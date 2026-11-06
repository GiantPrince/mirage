// SPDX-License-Identifier: Apache-2.0
// Uniform/Gumbel transform adapted from vLLM PR #51367 at
// 6af68f63030a7b27f8c07fad5b6237dcb3964e5c.
// Copyright contributors to the vLLM project (uniform/Gumbel transform).
#pragma once
#include "mirage/persistent_kernel/serving_config.h"
#include "mirage/persistent_kernel/tasks/common/bfloat16.h"
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cuda/atomic>
#include <cuda/std/functional>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

namespace mirage {
namespace serving {
__device__ inline uint32_t score_bits(float score) {
  uint32_t bits = __float_as_uint(score == 0.f ? 0.f : score);
  return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}
// Reverse token IDs make argmax prefer the lowest ID when scores tie.
__device__ inline uint64_t score_key(float score, uint32_t token) {
  return (uint64_t(score_bits(score)) << 32) | (0xffffffffu - token);
}
struct Maximum {
  __device__ uint64_t operator()(uint64_t a, uint64_t b) const {
    return a > b ? a : b;
  }
};

struct Draw {
  float score = -INFINITY, noise = 0;
  int token = 0;
};

// Compare score + temperature * noise without rounding a large common logit
// offset into the noise. This is equivalent to centered Gumbel-max, but needs
// neither a preliminary maximum nor a second vocabulary pass. Near
// cancellation, use FP64 to resolve the sign instead of depending on the
// reduction order.
struct ChooseDraw {
  float temperature;
  __device__ Draw operator()(Draw a, Draw b) const {
    if (!isfinite(a.score) || !isfinite(b.score)) {
      return score_key(a.score, a.token) >= score_key(b.score, b.token) ? a : b;
    }
    if (a.score == b.score) {
      if (a.noise != b.noise) {
        return a.noise > b.noise ? a : b;
      }
      return a.token < b.token ? a : b;
    }
    float delta = a.score - b.score;
    float shift = temperature * (b.noise - a.noise);
    float margin = delta - shift;
    // Eight FP32 epsilons conservatively cover the subtraction/product errors.
    if (fabsf(margin) > 0x1p-20f * (fabsf(delta) + fabsf(shift))) {
      return margin > 0 ? a : b;
    }
    double exact = (double(a.score) - b.score) +
                   double(temperature) * (double(a.noise) - b.noise);
    if (exact != 0) {
      return exact > 0 ? a : b;
    }
    return a.token < b.token ? a : b;
  }
};

constexpr int PARALLEL_SAMPLING_SHARED_BYTES = 16 * 1024;
// Reserve the final 2 KiB of shared memory for CUB reductions.
constexpr int CUB_STORAGE_OFFSET_BYTES = 14 * 1024;
static_assert(sizeof(int) == sizeof(float));

template <int Threads, typename T, typename Op>
__device__ inline T cub_block_reduce(T value, Op op) {
  using Reduce =
      cub::BlockReduce<T, Threads, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
  struct Storage {
    typename Reduce::TempStorage reduce;
    T aggregate;
  };
  static_assert(CUB_STORAGE_OFFSET_BYTES + sizeof(Storage) <=
                PARALLEL_SAMPLING_SHARED_BYTES);
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto &storage =
      *reinterpret_cast<Storage *>(sampling_shared + CUB_STORAGE_OFFSET_BYTES);
  T result = Reduce(storage.reduce).Reduce(value, op);
  if (threadIdx.x == 0) {
    storage.aggregate = result;
  }
  __syncthreads();
  result = storage.aggregate;
  __syncthreads();
  return result;
}

// Reconstruct (bits + 0.5) / 2^32 from two exact 16-bit conversions.
// Keeping all source bits is important at u -> 0, the winning Gumbel tail.
__device__ inline float sampling_uniform(uint32_t bits) {
  return float(bits >> 16) * 0x1p-16f +
         (float(bits & 0xffffu) + 0.5f) * 0x1p-32f;
}

__device__ inline float gumbel_from_uniform(float u) {
  // Degree-8 expansion of -log1p(-u). Unlike log(1-u), it preserves tiny u.
  // This is the polynomial and endpoint handling used by vLLM's FP32 path.
  float polynomial = 1.f / 8.f;
#pragma unroll
  for (int order = 7; order >= 1; --order) {
    polynomial = 1.f / order + u * polynomial;
  }
  float exponential =
      u < 0.25f ? u * polynomial : -logf(fmaxf(1.f - u, 0x1p-24f));
  return -logf(exponential);
}

// Philox subsequence = generation position, offset = token ID. A thread caches
// the most recent group of four: dense vector loads reuse all curand4 outputs;
// sparse candidate lists remain independent of traversal order and block size.
// The stream depends only on seed, position, and token, never the batch slot.
struct SamplingRng {
  uint64_t seed;
  uint32_t position;
  mutable uint32_t group = 0xffffffffu;
  mutable uint4 values{};
  __device__ SamplingRng(uint64_t seed, uint32_t position)
      : seed(seed), position(position) {}
  __device__ uint32_t bits(uint32_t token) const {
    if (group != token / 4) {
      curandStatePhilox4_32_10_t state;
      curand_init(seed, position, token & ~uint32_t(3), &state);
      values = curand4(&state);
      group = token / 4;
    }
    return (token & 3) == 0   ? values.x
           : (token & 3) == 1 ? values.y
           : (token & 3) == 2 ? values.z
                              : values.w;
  }
  __device__ float gumbel(uint32_t token) const {
    return gumbel_from_uniform(sampling_uniform(bits(token)));
  }
};

// A thread loads adjacent logits as a vector; the tail or an unaligned row
// falls back to scalar loads. Warp coalescing and per-thread width are
// separate.
template <typename T, int Width>
struct alignas(Width * sizeof(T)) SampleVector {
  T values[Width];
};

template <typename T, int Width>
__device__ inline SampleVector<T, Width>
    load_vector(T const *values, int start, int size) {
  if (start + Width <= size &&
      uintptr_t(values + start) % alignof(SampleVector<T, Width>) == 0) {
    return *reinterpret_cast<SampleVector<T, Width> const *>(values + start);
  }
  SampleVector<T, Width> result{};
#pragma unroll
  for (int i = 0; i < Width; ++i) {
    if (start + i < size) {
      result.values[i] = values[start + i];
    }
  }
  return result;
}

struct SamplingScratch {
  int *frequency_counts;
  uint32_t *seen_total;
  uint32_t *seen_generated;
  int *history_cursor;
  int bitset_words;

  __device__ SamplingScratch(float *scratch, int padded_vocab)
      : frequency_counts(reinterpret_cast<int *>(scratch)),
        bitset_words(sampling_bitset_words(padded_vocab)) {
    seen_total = reinterpret_cast<uint32_t *>(
        frequency_counts + sampling_frequency_words(padded_vocab));
    seen_generated = seen_total + bitset_words;
    history_cursor = reinterpret_cast<int *>(seen_generated + bitset_words +
                                             HISTORY_CURSOR_OFFSET_WORDS);
  }
};

__device__ inline void prepare_history(SamplingScratch const &scratch,
                                       ServingConfig const *cfg,
                                       long long const *history,
                                       int history_len,
                                       int prompt_len,
                                       int generation_position,
                                       int vocab,
                                       float frequency) {
  // Reuse only private row state initialized by this request's first sample.
  // Disabling caching rebuilds the histograms for each generated token.
  bool cached = cfg->cache_history && generation_position > 0;
  int begin = cached ? *scratch.history_cursor : 0;
  if (!cached) {
    if (frequency != 0.f) {
      for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
        scratch.frequency_counts[v] = 0;
      }
    }
    for (int v = threadIdx.x; v < scratch.bitset_words; v += blockDim.x) {
      scratch.seen_total[v] = 0;
      scratch.seen_generated[v] = 0;
    }
    __syncthreads();
  }
  for (int i = begin + threadIdx.x; i < history_len; i += blockDim.x) {
    long long token = history[i];
    if (token >= 0 && token < vocab) {
      atomicOr(scratch.seen_total + (token >> 5), uint32_t(1) << (token & 31));
      if (i >= prompt_len) {
        atomicOr(scratch.seen_generated + (token >> 5),
                 uint32_t(1) << (token & 31));
        if (frequency != 0.f) {
          atomicAdd(scratch.frequency_counts + token, 1);
        }
      }
    }
  }
  // All threads must consume the old cursor before it is updated.
  __syncthreads();
  if (threadIdx.x == 0) {
    *scratch.history_cursor = history_len;
  }
}

struct SamplingScoreReader {
  ServingConfig const *cfg;
  int const *frequency_counts;
  uint32_t const *seen_total;
  uint32_t const *seen_generated;
  float repetition, frequency, presence;
  int bias_count;

  __device__ float apply_bias(int v, float score) const {
    // Bias IDs are sorted by the host packer. Keep the small bias table in
    // ServingConfig instead of materializing a modified vocabulary row.
    if (bias_count) {
      int lo = 0, hi = bias_count;
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cfg->biases[mid].token < v) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      if (lo < bias_count && cfg->biases[lo].token == v) {
        score += float(cfg->biases[lo].value);
      }
    }
    return score;
  }

  __device__ float apply_generated_penalties(int v, float score) const {
    // Most vocabulary IDs never occur in the generated history. Avoid a
    // dense count-array read unless its bit is set.
    if ((frequency != 0.f || presence != 0.f) &&
        (seen_generated[v >> 5] & (uint32_t(1) << (v & 31)))) {
      if (frequency != 0.f) {
        score -= frequency * frequency_counts[v];
      }
      if (presence != 0.f) {
        score -= presence;
      }
    }
    return score;
  }

  __device__ float raw(int v, float score) const {
    score = apply_bias(v, score);
    if (repetition != 1.f && (seen_total[v >> 5] & (uint32_t(1) << (v & 31)))) {
      score = score > 0 ? score / repetition : score * repetition;
    }
    score = apply_generated_penalties(v, score);
    return isnan(score) ? -INFINITY : score;
  }
};

// Vector loads avoid a second logit read when penalties or biases are applied.
template <int Width = 16, typename T, typename Visit>
__device__ inline void scan_logits(T const *logits, int vocab, Visit visit) {
  for (int base = Width * threadIdx.x; base < vocab;
       base += Width * blockDim.x) {
    auto values = load_vector<T, Width>(logits, base, vocab);
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      int v = base + i;
      if (v < vocab) {
        visit(v, static_cast<float>(values.values[i]));
      }
    }
  }
}

enum ServingSamplingPhase {
  SAMPLE_PREPARE = 0,
  SAMPLE_SCORES = 1,
  SAMPLE_SELECT = 2,
  SAMPLE_RADIX_1 = 3,
  SAMPLE_RADIX_2 = 4,
  SAMPLE_RADIX_3 = 5,
  SAMPLE_DRAW = 6,
  SAMPLE_CHECK = 7,
  SAMPLE_RETRY = 8,
  SAMPLE_FINISH = 9,
};
// Counters join vocabulary partitions without blocking persistent workers.
// Radix counters are reused by mass checks after the cutoff is resolved.
struct ParallelSamplingState {
  int arrivals[3], radix_arrivals[4];
  int active, done, row, logit_row, position, vocab, parts;
  int k, best_token, token, radix_rank;
  uint32_t radix_prefix, radix_mask;
  bool radix_done, compact_top_k, pivot_top_p, needs_check, refine_top_k;
  int prefilter_bucket;
  float maximum, temperature, candidate_weight, candidate_score;
  double p, target;
};
struct ParallelSamplingPart {
  uint64_t best, second;
  Draw draw;
  double above_mass, total_mass;
  int candidate_count;
  uint32_t radix_counts[256];
};
static_assert(sizeof(ParallelSamplingState) <=
              PARALLEL_SAMPLING_STATE_WORDS * sizeof(float));
static_assert(sizeof(ParallelSamplingPart) <=
              PARALLEL_SAMPLING_PART_WORDS * sizeof(float));
constexpr int PARALLEL_CUB_OFFSET = 12 * 1024;
struct ParallelSamplingScratch {
  ParallelSamplingState &state;
  // Layout: request state, candidate IDs, then one state slab per partition.
  // Candidate IDs are partition-local offsets when they fit in 16 bits.
  int *candidate_storage;
  bool compact_indices;
  float *partials; // Per-partition ParallelSamplingPart storage.

  __device__ ParallelSamplingScratch(float *buffer, int vocab, int parts)
      : state(*reinterpret_cast<ParallelSamplingState *>(buffer)),
        candidate_storage(
            reinterpret_cast<int *>(buffer + PARALLEL_SAMPLING_STATE_WORDS)),
        compact_indices(parallel_sampling_compact_indices(vocab, parts)),
        partials(buffer + PARALLEL_SAMPLING_STATE_WORDS +
                 parallel_sampling_index_words(vocab, parts)) {}

  __device__ int candidate_token(int begin, int index) const {
    if (compact_indices) {
      auto *offsets = reinterpret_cast<uint16_t const *>(candidate_storage);
      return begin + offsets[begin + index];
    }
    return candidate_storage[begin + index];
  }

  __device__ void store_candidate(int begin, int index, int token) const {
    if (compact_indices) {
      auto *offsets = reinterpret_cast<uint16_t *>(candidate_storage);
      offsets[begin + index] = uint16_t(token - begin);
    } else {
      candidate_storage[begin + index] = token;
    }
  }

  __device__ ParallelSamplingPart &part(int index) const {
    return *reinterpret_cast<ParallelSamplingPart *>(
        partials + index * PARALLEL_SAMPLING_PART_WORDS);
  }
};

// Reuse the idle frequency-count row for Gumbel noise. Frequency penalties
// keep their count row and regenerate noise from the same per-token RNG.
template <typename T>
struct ParallelSamplingReader {
  T const *logits;
  SamplingScoreReader score_reader;
  ParallelSamplingState const &state;
  float *cached_noise;
  bool plain_logits;

  __device__ static SamplingScoreReader make_score_reader(
      float *history_buffer, int padded_vocab, ServingConfig const *cfg) {
    SamplingScratch history(history_buffer, padded_vocab);
    return {cfg,
            history.frequency_counts,
            history.seen_total,
            history.seen_generated,
            float(cfg->repetition_penalty),
            float(cfg->frequency_penalty),
            float(cfg->presence_penalty),
            int(cfg->bias_count)};
  }

  __device__ ParallelSamplingReader(T const *logits,
                                    float *history_buffer,
                                    int padded_vocab,
                                    ServingConfig const *cfg,
                                    ParallelSamplingState const &state)
      : logits(logits),
        score_reader(make_score_reader(history_buffer, padded_vocab, cfg)),
        state(state),
        cached_noise(cfg->frequency_penalty == 0 ? history_buffer : nullptr),
        plain_logits(cfg->bias_count == 0 && cfg->repetition_penalty == 1 &&
                     cfg->frequency_penalty == 0 &&
                     cfg->presence_penalty == 0) {}

  __device__ float processed_score(int token, float value) const {
    return plain_logits ? (isnan(value) ? -INFINITY : value)
                        : score_reader.raw(token, value);
  }
  __device__ float processed_score(int token) const {
    return processed_score(token, float(logits[token]));
  }
  __device__ float scaled_score(int token) const {
    return (processed_score(token) - state.maximum) / state.temperature;
  }
  __device__ float weight(int token) const {
    float score = processed_score(token);
    if (state.radix_done && score_bits(score) < state.radix_prefix) {
      return 0.f;
    }
    return expf((score - state.maximum) / state.temperature);
  }
};

// No peer waits inside a worker. Dependency events separate task phases.
__device__ inline bool parallel_last_arrival(ParallelSamplingState &state,
                                             int phase) {
  __syncthreads();
  bool last = false;
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> counter(
        state.arrivals[phase]);
    last = counter.fetch_add(1, cuda::memory_order_acq_rel) == state.parts - 1;
  }
  return __syncthreads_or(last);
}

// Phase 0: initialize one request and its history-dependent penalties.
__device__ inline void parallel_prepare(ParallelSamplingScratch const &scratch,
                                        float *history_buffer,
                                        ServingConfig const *cfg,
                                        long long const *history_tokens,
                                        int history_len,
                                        int prompt,
                                        int position,
                                        int row,
                                        int logit_row,
                                        int padded_vocab,
                                        int parts,
                                        bool active) {
  auto &s = scratch.state;
  if (threadIdx.x == 0) {
    s = ParallelSamplingState{};
    s.active = active;
    s.row = row;
    s.logit_row = logit_row;
    s.position = position;
    s.parts = parts;
    s.token = INT_MAX;
    if (active) {
      s.vocab = min(padded_vocab, int(cfg->vocab_size));
      s.k = cfg->top_k > 0 && cfg->top_k < s.vocab ? int(cfg->top_k) : 0;
      s.radix_rank = s.k;
      s.temperature = fmaxf(float(cfg->temperature), 0x1p-126f);
      s.p = cfg->top_p;
      s.pivot_top_p =
          !s.k && cfg->top_k != 1 && cfg->temperature > 0 && s.p < 1;
    }
  }
  __syncthreads();
  if (active && (cfg->frequency_penalty != 0 || cfg->presence_penalty != 0 ||
                 cfg->repetition_penalty != 1)) {
    SamplingScratch history(history_buffer, padded_vocab);
    prepare_history(history,
                    cfg,
                    history_tokens,
                    history_len,
                    prompt,
                    position,
                    s.vocab,
                    float(cfg->frequency_penalty));
  }
}

struct TopTwo {
  uint64_t first, second;
};
struct ChooseTopTwo {
  __device__ TopTwo operator()(TopTwo a, TopTwo b) const {
    if (b.first > a.first) {
      return {b.first, max(a.first, b.second)};
    }
    return {a.first, max(a.second, b.first)};
  }
};

// Phase 1: process logits and find maxima, a safe top-k score bound, or the
// first Gumbel proposal. Greedy and unfiltered draws finish here.
template <int Threads, typename T>
__device__ __noinline__ void
    parallel_scores(T const *logits,
                    float *history_scratch,
                    ParallelSamplingScratch const &scratch,
                    ServingConfig const *cfg,
                    int partition,
                    int padded_vocab) {
  auto &s = scratch.state;
  if (!s.active || s.done) {
    return;
  }
  ParallelSamplingReader reader(logits, history_scratch, padded_vocab, cfg, s);
  bool greedy = cfg->temperature == 0 || cfg->top_k == 1;
  bool unfiltered = !greedy && !s.k && s.p == 1;
  bool pivot_top_p = !greedy && !s.k && s.p < 1;
  bool draw_token = unfiltered || pivot_top_p;
  int begin = s.vocab * partition / s.parts,
      end = s.vocab * (partition + 1) / s.parts;
  uint64_t best = 0, second = 0;
  // Each partition supplies at most two distinct scores to the bound.
  bool compact_top_k = !greedy && s.k && s.k <= 2 * s.parts;
  Draw draw;
  ChooseDraw choose{s.temperature};
  SamplingRng rng{uint64_t(cfg->seed), uint32_t(s.position)};
  scan_logits<4>(logits + begin, end - begin, [&](int offset, float logit) {
    int v = begin + offset;
    float score = reader.processed_score(v, logit);
    uint64_t key = score_key(score, v);
    if (compact_top_k && s.k > s.parts) {
      second = key > best ? best : max(second, key);
    }
    best = key > best ? key : best;
    if (draw_token) {
      float noise = isfinite(score) ? rng.gumbel(v) : 0.f;
      if (pivot_top_p && reader.cached_noise) {
        reader.cached_noise[v] = noise;
      }
      draw = choose(draw, Draw{score, noise, v});
    }
  });
  if (compact_top_k && s.k > s.parts) {
    auto top = cub_block_reduce<Threads>(TopTwo{best, second}, ChooseTopTwo{});
    best = top.first;
    second = top.second;
  } else {
    best = cub_block_reduce<Threads>(best, Maximum{});
  }
  if (draw_token) {
    draw = cub_block_reduce<Threads>(draw, choose);
  }
  if (threadIdx.x == 0) {
    scratch.part(partition).best = best;
    scratch.part(partition).second = second;
    scratch.part(partition).draw = draw;
  }
  if (!parallel_last_arrival(s, 1)) {
    return;
  }
  int partial_count = s.k <= s.parts ? s.parts : 2 * s.parts;
  // The kth score in this subset is a safe lower bound on the global
  // kth score, even when the largest logits cluster in one partition.
  if (compact_top_k && partial_count <= 32) {
    if (threadIdx.x < 32) {
      int lane = threadIdx.x;
      uint64_t key = 0;
      if (lane < partial_count) {
        int part = s.k <= s.parts ? lane : lane / 2;
        key = s.k <= s.parts || lane % 2 == 0 ? scratch.part(part).best
                                              : scratch.part(part).second;
      }
      for (int group = 2; group <= 32; group *= 2) {
        for (int gap = group / 2; gap; gap /= 2) {
          uint64_t other = __shfl_xor_sync(0xffffffffu, key, gap);
          bool larger = ((lane & gap) == 0) != ((lane & group) != 0);
          key = larger ? max(key, other) : min(key, other);
        }
      }
      if (lane == 0) {
        s.best_token = 0xffffffffu - uint32_t(key);
        s.maximum = reader.processed_score(s.best_token);
        if (!isfinite(s.maximum)) {
          s.token = s.best_token;
          s.done = true;
        }
      }
      if (lane == s.k - 1) {
        s.candidate_score =
            key ? reader.processed_score(0xffffffffu - uint32_t(key))
                : -INFINITY;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      s.compact_top_k = isfinite(s.candidate_score);
    }
    return;
  }
  if (threadIdx.x == 0) {
    best = 0;
    uint64_t pivot = UINT64_MAX;
    Draw winner;
    for (int j = 0; j < s.parts; ++j) {
      auto &part = scratch.part(j);
      if (compact_top_k) {
        pivot = min(pivot, s.k > s.parts ? part.second : part.best);
      }
      if (part.best > best) {
        best = part.best;
      }
      if (draw_token) {
        winner = choose(winner, part.draw);
      }
    }
    s.best_token = 0xffffffffu - uint32_t(best);
    s.maximum = reader.processed_score(s.best_token);
    if (compact_top_k) {
      int token = 0xffffffffu - uint32_t(pivot);
      s.candidate_score = pivot ? reader.processed_score(token) : -INFINITY;
      s.compact_top_k = isfinite(s.candidate_score);
    }
    if (pivot_top_p) {
      s.token = winner.token;
      s.candidate_weight =
          expf((winner.score - s.maximum) / s.temperature);
      s.needs_check = true;
    }
    if (greedy || !isfinite(s.maximum) || unfiltered) {
      s.token = unfiltered ? winner.token : s.best_token;
      s.done = true;
    }
  }
}

template <int Threads>
__device__ inline int const *
    parallel_candidate_offsets(ParallelSamplingScratch const &scratch) {
  auto &s = scratch.state;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto *offsets = reinterpret_cast<int *>(sampling_shared + 14 * 1024);
  if (threadIdx.x < 32) {
    int lane = threadIdx.x;
    int count = lane < s.parts ? scratch.part(lane).candidate_count : 0;
    for (int delta = 1; delta < 32; delta *= 2) {
      int value = __shfl_up_sync(0xffffffffu, count, delta);
      if (lane >= delta) {
        count += value;
      }
    }
    if (lane < s.parts) {
      offsets[lane + 1] = count;
    }
    if (lane == 0) {
      offsets[0] = 0;
    }
  }
  __syncthreads();
  return offsets;
}

// The vocabulary scan is distributed; only this bounded candidate set
// needs a final warp-local sort and categorical draw.
// Its capacity is 32 lanes times two keys per lane, not a tuning threshold.
template <typename T>
__device__ inline void
    parallel_sort_warp(ParallelSamplingScratch const &scratch,
                       ParallelSamplingReader<T> const &reader,
                       ServingConfig const *cfg,
                       int const *offsets) {
  auto &s = scratch.state;
  int lane = threadIdx.x;
  int token_bits = 32 - __clz(uint32_t(s.vocab - 1));
  uint32_t token_mask = (uint32_t(1) << token_bits) - 1;
  uint32_t minimum = score_bits(nextafterf(s.candidate_score, -INFINITY));
  uint64_t keys[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int index = lane * 2 + i;
    keys[i] = uint64_t(minimum) << token_bits;
    if (index < offsets[s.parts]) {
      int lo = 0, hi = s.parts;
      while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (offsets[mid] <= index) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      int begin = s.vocab * lo / s.parts;
      int token = scratch.candidate_token(begin, index - offsets[lo]);
      keys[i] =
          (uint64_t(score_bits(reader.processed_score(token))) << token_bits) |
          (token_mask - uint32_t(token));
    }
  }
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  using Sort = cub::WarpMergeSort<uint64_t, 2>;
  auto &storage =
      *reinterpret_cast<typename Sort::TempStorage *>(sampling_shared);
  Sort(storage).Sort(keys, cuda::std::greater<uint64_t>{});
  int selected = s.k - 1;
  uint32_t cutoff = __shfl_sync(
      0xffffffffu, uint32_t(keys[selected % 2] >> token_bits), selected / 2);
  double weights[2];
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    int token = token_mask - uint32_t(keys[i] & token_mask);
    weights[i] = lane * 2 + i < offsets[s.parts] &&
                         uint32_t(keys[i] >> token_bits) >= cutoff
                     ? reader.weight(token)
                     : 0;
  }
  double local = weights[0] + weights[1], prefix = local;
  for (int delta = 1; delta < 32; delta *= 2) {
    double value = __shfl_up_sync(0xffffffffu, prefix, delta);
    if (lane >= delta) {
      prefix += value;
    }
  }
  double cdf[2] = {prefix - local + weights[0], prefix};
  double total = __shfl_sync(0xffffffffu, prefix, 31);
  if (s.p < 1) {
    double target = s.p * total;
    uint32_t boundary = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (weights[i] > 0 && cdf[i] >= target && cdf[i] - weights[i] < target) {
        boundary = max(boundary, __float_as_uint(float(weights[i])));
      }
    }
    boundary = __reduce_max_sync(0xffffffffu, boundary);
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      if (__float_as_uint(float(weights[i])) < boundary) {
        weights[i] = 0;
      }
    }
    local = weights[0] + weights[1];
    prefix = local;
    for (int delta = 1; delta < 32; delta *= 2) {
      double value = __shfl_up_sync(0xffffffffu, prefix, delta);
      if (lane >= delta) {
        prefix += value;
      }
    }
    cdf[0] = prefix - local + weights[0];
    cdf[1] = prefix;
    total = __shfl_sync(0xffffffffu, prefix, 31);
  }
  double uniform = 0;
  if (lane == 0) {
    SamplingRng rng{uint64_t(cfg->seed), uint32_t(s.position)};
    uniform = (double(rng.bits(0)) + .5) * 0x1p-32;
  }
  double target = __shfl_sync(0xffffffffu, uniform, 0) * total;
  int token = INT_MAX;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (weights[i] > 0 && cdf[i] > target && cdf[i] - weights[i] <= target) {
      token = min(token, int(token_mask - uint32_t(keys[i] & token_mask)));
    }
  }
  token = __reduce_min_sync(0xffffffffu, token);
  if (lane == 0) {
    s.token = token == INT_MAX ? s.best_token : token;
    s.done = true;
  }
}

template <int Threads, typename T>
__device__ inline void
    parallel_sample_candidates(ParallelSamplingScratch const &scratch,
                               ParallelSamplingReader<T> const &reader,
                               ServingConfig const *cfg,
                               int const *offsets) {
  if (threadIdx.x < 32) {
    parallel_sort_warp(scratch, reader, cfg, offsets);
  }
  __syncthreads();
}

template <int Threads, typename T>
__device__ inline void
    parallel_compact_top_k(ParallelSamplingScratch const &scratch,
                           ParallelSamplingReader<T> const &reader,
                           ServingConfig const *cfg,
                           int partition) {
  auto &s = scratch.state;
  auto &part = scratch.part(partition);
  constexpr int Bins = 32, Warps = Threads / 32;
  static_assert(Bins * PARALLEL_SAMPLING_MAX_PARTS * sizeof(uint32_t) <=
                PARALLEL_CUB_OFFSET);
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto *histogram = reinterpret_cast<uint32_t *>(sampling_shared);
  for (int i = threadIdx.x; i < Bins * Warps; i += Threads) {
    histogram[i] = 0;
  }
  if (threadIdx.x == 0) {
    part.candidate_count = 0;
  }
  __syncthreads();
  int begin = s.vocab * partition / s.parts,
      end = s.vocab * (partition + 1) / s.parts;
  int lane = threadIdx.x % 32, warp_offset = threadIdx.x / 32 * Bins;
  float pivot = s.candidate_score;
  uint32_t lower = score_bits(pivot);
  int shift = max(0, 27 - __clz(score_bits(s.maximum) - lower));
  for (int token = begin + threadIdx.x; token < end; token += Threads) {
    float score = reader.processed_score(token);
    bool keep = score >= pivot;
    unsigned active = __activemask();
    unsigned selected = __ballot_sync(active, keep);
    if (selected) {
      int offset = 0;
      if (lane == 0) {
        offset = atomicAdd(&part.candidate_count, __popc(selected));
      }
      offset = __shfl_sync(active, offset, 0);
      if (keep) {
        scratch.store_candidate(
            begin,
            offset + __popc(selected & ((uint32_t(1) << lane) - 1)),
            token);
        atomicAdd(histogram + warp_offset +
                      int((score_bits(score) - lower) >> shift),
                  1u);
      }
    }
  }
  __syncthreads();
  for (int bin = threadIdx.x; bin < Bins; bin += Threads) {
    uint32_t count = 0;
    for (int warp = 0; warp < Warps; ++warp) {
      count += histogram[warp * Bins + bin];
    }
    part.radix_counts[bin] = count;
  }
  if (!parallel_last_arrival(s, 2)) {
    return;
  }
  int const *offsets = parallel_candidate_offsets<Threads>(scratch);
  if (offsets[s.parts] <= 64) {
    parallel_sample_candidates<Threads>(scratch, reader, cfg, offsets);
    return;
  }
  // Cache the small relative-score histograms before combining them.
  // This narrows skewed candidate lists without rereading the vocabulary.
  auto *cached = reinterpret_cast<uint32_t *>(sampling_shared);
  for (int index = threadIdx.x; index < s.parts * Bins; index += Threads) {
    cached[index] = scratch.part(index / Bins).radix_counts[index % Bins];
  }
  __syncthreads();
  int bin = Bins - 1 - int(threadIdx.x), count = 0;
  if (bin >= 0) {
    for (int j = 0; j < s.parts; ++j) {
      count += cached[j * Bins + bin];
    }
  }
  using Scan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  auto &storage = *reinterpret_cast<typename Scan::TempStorage *>(
      sampling_shared + PARALLEL_CUB_OFFSET);
  int cumulative;
  Scan(storage).InclusiveSum(count, cumulative);
  if (bin >= 0 && cumulative >= s.k && cumulative - count < s.k) {
    s.prefilter_bucket = bin;
    s.refine_top_k = cumulative <= 64;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    s.radix_rank = s.k;
    s.radix_done = false;
    s.radix_prefix = 0;
    s.radix_mask = 0;
    // Reuse the join counter only if the next phase compacts again.
    if (s.refine_top_k) {
      s.arrivals[2] = 0;
    }
  }
}

template <int Threads, typename T>
__device__ inline void
    parallel_refine_top_k(ParallelSamplingScratch const &scratch,
                          ParallelSamplingReader<T> const &reader,
                          ServingConfig const *cfg,
                          int partition) {
  auto &s = scratch.state;
  auto &part = scratch.part(partition);
  int begin = s.vocab * partition / s.parts, count = part.candidate_count,
      offset = 0;
  uint32_t lower = score_bits(s.candidate_score);
  int shift = max(0, 27 - __clz(score_bits(s.maximum) - lower));
  int boundary = s.prefilter_bucket;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  using Scan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  auto &storage =
      *reinterpret_cast<typename Scan::TempStorage *>(sampling_shared);
  for (int base = 0; base < count; base += Threads * 4) {
    int tokens[4];
    unsigned selected = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      int index = base + int(threadIdx.x) * 4 + i;
      if (index < count) {
        tokens[i] = scratch.candidate_token(begin, index);
        if (int((score_bits(reader.processed_score(tokens[i])) - lower) >>
                shift) >= boundary) {
          selected |= 1u << i;
        }
      }
    }
    // Read the whole tile before compacting into the same index array.
    __syncthreads();
    int prefix, total;
    Scan(storage).ExclusiveSum(__popc(selected), prefix, total);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (selected & (1u << i)) {
        scratch.store_candidate(begin, offset + prefix++, tokens[i]);
      }
    }
    offset += total;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part.candidate_count = offset;
  }
  if (!parallel_last_arrival(s, 2)) {
    return;
  }
  int const *offsets = parallel_candidate_offsets<Threads>(scratch);
  parallel_sample_candidates<Threads>(scratch, reader, cfg, offsets);
}

// A radix pass counts one score byte across the vocabulary partitions. The
// second pass compacts the first selected byte for subsequent refinement.
template <int Threads, int Round, typename T>
__device__ inline void parallel_radix(ParallelSamplingScratch const &scratch,
                                      ParallelSamplingReader<T> const &reader,
                                      int partition) {
  auto &s = scratch.state;
  if (s.radix_done) {
    return;
  }
  constexpr int Bins = 256, Warps = Threads / 32, Shift = 24 - 8 * Round;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto *histogram = reinterpret_cast<uint32_t *>(sampling_shared);
  for (int i = threadIdx.x; i < Bins * Warps; i += Threads) {
    histogram[i] = 0;
  }
  auto &part = scratch.part(partition);
  if constexpr (Round == 1) {
    if (threadIdx.x == 0 && !s.compact_top_k) {
      part.candidate_count = 0;
    }
  }
  __syncthreads();
  int begin = s.vocab * partition / s.parts,
      end = s.vocab * (partition + 1) / s.parts;
  int count =
      (Round <= 1 && !s.compact_top_k) ? end - begin : part.candidate_count;
  uint32_t prefix = s.radix_prefix, mask = s.radix_mask;
  int lane = threadIdx.x % 32, warp_offset = threadIdx.x / 32 * Bins;
  for (int i = threadIdx.x; i < count; i += Threads) {
    int token = (Round <= 1 && !s.compact_top_k)
                    ? begin + i
                    : scratch.candidate_token(begin, i);
    uint32_t bits = score_bits(reader.processed_score(token));
    bool keep = (bits & mask) == prefix;
    if constexpr (Round == 1) {
      if (!s.compact_top_k) {
        unsigned active = __activemask();
        unsigned selected = __ballot_sync(active, keep);
        int base = 0;
        if (lane == 0 && selected) {
          base = atomicAdd(&part.candidate_count, __popc(selected));
        }
        base = __shfl_sync(active, base, 0);
        if (keep) {
          scratch.store_candidate(
              begin,
              base + __popc(selected & ((uint32_t(1) << lane) - 1)),
              token);
        }
      }
    }
    if (keep) {
      atomicAdd(histogram + warp_offset + ((bits >> Shift) & 255), 1u);
    }
  }
  __syncthreads();
  for (int bin = threadIdx.x; bin < Bins; bin += Threads) {
    uint32_t count = 0;
    for (int warp = 0; warp < Warps; ++warp) {
      count += histogram[warp * Bins + bin];
    }
    part.radix_counts[bin] = count;
  }
  __syncthreads();
  bool last = false;
  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> counter(
        s.radix_arrivals[Round]);
    last = counter.fetch_add(1, cuda::memory_order_acq_rel) == s.parts - 1;
  }
  if (!__syncthreads_or(last)) {
    return;
  }
  using Scan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  auto &storage = *reinterpret_cast<typename Scan::TempStorage *>(
      sampling_shared + PARALLEL_CUB_OFFSET);
  int counts[2], cdf[2];
  for (int i = 0; i < 2; ++i) {
    int bin = 255 - int(threadIdx.x) * 2 - i;
    counts[i] = 0;
    if (bin >= 0) {
      for (int j = 0; j < s.parts; ++j) {
        counts[i] += scratch.part(j).radix_counts[bin];
      }
    }
  }
  Scan(storage).InclusiveSum(counts, cdf);
  int rank = s.radix_rank;
  for (int i = 0; i < 2; ++i) {
    int bin = 255 - int(threadIdx.x) * 2 - i;
    if (bin >= 0 && cdf[i] >= rank && cdf[i] - counts[i] < rank) {
      s.radix_prefix |= uint32_t(bin) << Shift;
      s.radix_mask |= 255u << Shift;
      s.radix_rank = rank - (cdf[i] - counts[i]);
      // Plain BF16 logits have no remaining score bits after the second byte.
      if (s.radix_rank == counts[i] || Round == 3 ||
          (Round == 1 &&
           (std::is_same<T, __nv_bfloat16>::value ||
            std::is_same<T, ::type::bfloat16_t>::value) &&
           reader.plain_logits)) {
        s.radix_done = true;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && s.radix_done) {
    for (int j = 0; j < 4; ++j) {
      s.radix_arrivals[j] = 0;
    }
  }
}

// Every retry uses the same per-token Gumbels. Discarding a rejected score
// pivot therefore preserves the largest perturbed score in the nucleus.
template <int Threads, bool Initial, typename T>
__device__ inline void
    parallel_pivot_draw(ParallelSamplingScratch const &scratch,
                        ParallelSamplingReader<T> const &reader,
                        ServingConfig const *cfg,
                        int partition,
                        bool partial) {
  auto &s = scratch.state;
  int begin = partial ? s.vocab * partition / s.parts : 0;
  int end = partial ? s.vocab * (partition + 1) / s.parts : s.vocab;
  float maximum = s.maximum, temperature = s.temperature,
        cutoff = Initial ? 0 : s.candidate_weight;
  uint32_t prefix = s.radix_prefix;
  bool filtered = s.radix_done;
  Draw draw;
  ChooseDraw choose{temperature};
  SamplingRng rng{uint64_t(cfg->seed), uint32_t(s.position)};
  scan_logits<4>(
      reader.logits + begin, end - begin, [&](int offset, float value) {
        int token = begin + offset;
        float score = reader.processed_score(token, value);
        float weight = filtered && score_bits(score) < prefix
                           ? 0
                           : expf((score - maximum) / temperature);
        if (weight > cutoff) {
          float noise = !Initial && reader.cached_noise
                            ? reader.cached_noise[token]
                            : rng.gumbel(token);
          if constexpr (Initial) {
            if (reader.cached_noise) {
              reader.cached_noise[token] = noise;
            }
          }
          draw = choose(draw, Draw{score, noise, token});
        }
      });
  draw = cub_block_reduce<Threads>(draw, choose);
  if (partial) {
    if (threadIdx.x == 0) {
      scratch.part(partition).draw = draw;
    }
    if (!parallel_last_arrival(s, 0)) {
      return;
    }
    if (threadIdx.x == 0) {
      Draw best;
      for (int j = 0; j < s.parts; ++j) {
        best = choose(best, scratch.part(j).draw);
      }
      s.token = best.token;
      s.candidate_weight = reader.weight(best.token);
      s.needs_check = true;
      s.arrivals[0] = 0;
      if constexpr (Initial) {
        if (s.p == 1) {
          s.done = true;
        }
      }
    }
  } else if (threadIdx.x == 0) {
    s.token = draw.token;
    s.candidate_weight = reader.weight(draw.token);
    s.needs_check = true;
  }
  __syncthreads();
}
// A candidate belongs to the tie-inclusive nucleus exactly when the mass
// strictly above its weight is smaller than p times the retained total.
template <int Threads, int Round, typename T>
__device__ inline void
    parallel_pivot_check(ParallelSamplingScratch const &scratch,
                         ParallelSamplingReader<T> const &reader,
                         int partition,
                         bool partial) {
  auto &s = scratch.state;
  int begin = partial ? s.vocab * partition / s.parts : 0;
  int end = partial ? s.vocab * (partition + 1) / s.parts : s.vocab;
  double mass = 0, total = 0;
  float cutoff = s.candidate_weight, maximum = s.maximum,
        temperature = s.temperature;
  bool filtered = s.radix_done;
  uint32_t prefix = s.radix_prefix;
  scan_logits<4>(
      reader.logits + begin, end - begin, [&](int offset, float value) {
        int token = begin + offset;
        float score = reader.processed_score(token, value);
        float weight = filtered && score_bits(score) < prefix
                           ? 0
                           : expf((score - maximum) / temperature);
        if constexpr (Round == 0) {
          total += double(weight);
        }
        if (weight > cutoff) {
          mass += double(weight);
        }
      });
  mass = cub_block_reduce<Threads>(mass, cuda::std::plus<double>{});
  if constexpr (Round == 0) {
    total = cub_block_reduce<Threads>(total, cuda::std::plus<double>{});
  }
  if (partial) {
    if (threadIdx.x == 0) {
      scratch.part(partition).above_mass = mass;
      scratch.part(partition).total_mass = total;
    }
    __syncthreads();
    bool last = false;
    if (threadIdx.x == 0) {
      cuda::atomic_ref<int, cuda::thread_scope_device> arrivals(
          s.radix_arrivals[Round]);
      last = arrivals.fetch_add(1, cuda::memory_order_acq_rel) == s.parts - 1;
    }
    if (!__syncthreads_or(last)) {
      return;
    }
    if (threadIdx.x == 0) {
      mass = 0;
      total = 0;
      for (int j = 0; j < s.parts; ++j) {
        mass += scratch.part(j).above_mass;
        total += scratch.part(j).total_mass;
      }
      if constexpr (Round == 0) {
        s.target = s.p * total;
      }
      s.needs_check = false;
      if (mass < s.target) {
        s.done = true;
      }
    }
  } else if (threadIdx.x == 0) {
    s.needs_check = false;
    if (mass < s.target) {
      s.done = true;
    }
  }
  __syncthreads();
}

// A compact-list overflow shifts radix and the first draw/check by one
// phase. Dependency edges provide each cross-CTA synchronization point.
template <int Threads, int Phase, typename T>
__device__ inline void parallel_filter(ParallelSamplingScratch const &scratch,
                                       ParallelSamplingReader<T> const &reader,
                                       ServingConfig const *cfg,
                                       int partition) {
  auto &s = scratch.state;
  if (!s.active || s.done) {
    return;
  }
  if (s.pivot_top_p) {
    if constexpr (Phase % 2 == 0) {
      parallel_pivot_check<Threads, (Phase - 2) / 2>(
          scratch, reader, partition, true);
    } else {
      parallel_pivot_draw<Threads, false>(
          scratch, reader, cfg, partition, true);
    }
    return;
  }
  if constexpr (Phase == 2) {
    if (s.compact_top_k) {
      parallel_compact_top_k<Threads>(scratch, reader, cfg, partition);
    } else {
      parallel_radix<Threads, 0>(scratch, reader, partition);
    }
  } else if constexpr (Phase == 3) {
    if (s.compact_top_k) {
      if (s.refine_top_k) {
        parallel_refine_top_k<Threads>(scratch, reader, cfg, partition);
      } else {
        parallel_radix<Threads, 0>(scratch, reader, partition);
      }
    } else {
      parallel_radix<Threads, 1>(scratch, reader, partition);
    }
  } else if constexpr (Phase == 4) {
    if (s.compact_top_k) {
      parallel_radix<Threads, 1>(scratch, reader, partition);
    } else {
      parallel_radix<Threads, 2>(scratch, reader, partition);
    }
  } else if constexpr (Phase == 5) {
    if (s.compact_top_k) {
      parallel_radix<Threads, 2>(scratch, reader, partition);
    } else {
      parallel_radix<Threads, 3>(scratch, reader, partition);
    }
  } else if constexpr (Phase == 6) {
    if (s.compact_top_k) {
      parallel_radix<Threads, 3>(scratch, reader, partition);
    } else {
      parallel_pivot_draw<Threads, true>(scratch, reader, cfg, partition, true);
    }
  } else if constexpr (Phase == 7) {
    if (s.compact_top_k) {
      parallel_pivot_draw<Threads, true>(scratch, reader, cfg, partition, true);
    } else {
      parallel_pivot_check<Threads, 0>(scratch, reader, partition, true);
    }
  } else if constexpr (Phase == 8) {
    if (s.compact_top_k) {
      parallel_pivot_check<Threads, 0>(scratch, reader, partition, true);
    } else {
      parallel_pivot_draw<Threads, false>(
          scratch, reader, cfg, partition, true);
    }
  }
}

template <int Threads, typename T>
__device__ inline void parallel_finish(ParallelSamplingScratch const &scratch,
                                       ParallelSamplingReader<T> const &reader,
                                       long long *output,
                                       ServingConfig const *cfg) {
  auto &s = scratch.state;
  if (!s.active) {
    return;
  }
  // Fixed task phases handle the common case. Finish any remaining
  // refinement exactly; an iteration limit must not widen the nucleus.
  while (!s.done) {
    if (s.needs_check) {
      parallel_pivot_check<Threads, 1>(scratch, reader, 0, false);
    } else {
      parallel_pivot_draw<Threads, false>(scratch, reader, cfg, 0, false);
    }
  }
  if (threadIdx.x == 0) {
    output[s.logit_row] = s.token;
  }
}
#if defined(MODE_ONLINE_PINNED)
template <int Threads, int Phase, typename T>
__device__ inline void
    sample_request(T const *logits,
                   float *history_buffer,
                   float *buffer,
                   long long *output,
                   int padded_vocab,
                   int parts,
                   int slot,
                   int partition,
                   mirage::runtime::RuntimeConfig const &config) {
  ParallelSamplingScratch scratch(
      buffer + slot * parallel_sampling_scratch_words(padded_vocab, parts),
      padded_vocab,
      parts);
  auto &s = scratch.state;
  if constexpr (Phase == SAMPLE_PREPARE) {
    int row = config.request_ids[slot];
    int start = config.qo_indptr_buffer[slot],
        end = config.qo_indptr_buffer[slot + 1];
    bool active = row >= 0 && end > start;
    int prompt = active ? config.prompt_length[row] : 0;
    int position = active ? config.step[row] + end - start - 1 : 0;
    active = active && position + 1 >= prompt;
    parallel_prepare(scratch,
                     history_buffer + (active ? row : 0) *
                                          sampling_scratch_words(padded_vocab),
                     active ? config.generation_config + row : nullptr,
                     active ? config.tokens + row * MPK_MAX_SEQ_LENGTH
                            : nullptr,
                     position + 1,
                     prompt,
                     position + 1 - prompt,
                     row,
                     end - 1,
                     padded_vocab,
                     parts,
                     active);
  } else {
    if (!s.active || (Phase < SAMPLE_FINISH && partition >= s.parts)) {
      return;
    }
    auto *cfg = config.generation_config + s.row;
    auto *row_logits = logits + s.logit_row * padded_vocab;
    auto *row_history =
        history_buffer + s.row * sampling_scratch_words(padded_vocab);
    if constexpr (Phase == SAMPLE_SCORES) {
      parallel_scores<Threads>(
          row_logits, row_history, scratch, cfg, partition, padded_vocab);
    } else {
      ParallelSamplingReader reader(
          row_logits, row_history, padded_vocab, cfg, s);
      if constexpr (Phase == SAMPLE_FINISH) {
        parallel_finish<Threads>(scratch, reader, output, cfg);
      } else {
        parallel_filter<Threads, Phase>(scratch, reader, cfg, partition);
      }
    }
  }
}

#endif
} // namespace serving
} // namespace mirage
