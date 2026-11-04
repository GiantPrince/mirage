// SPDX-License-Identifier: Apache-2.0
// Uniform/Gumbel transform adapted from vLLM PR #51367 at
// 6af68f63030a7b27f8c07fad5b6237dcb3964e5c.
// Copyright contributors to the vLLM project (uniform/Gumbel transform).
#pragma once
#include "mirage/persistent_kernel/serving_config.h"
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
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

struct MaximumIndex {
  __device__ int operator()(int a, int b) const {
    return max(a, b);
  }
};

// One request runs through these task phases. Partitioned phases use one CTA
// per vocabulary partition and share a request-local scratch row.
enum ServingSamplingPhase {
  SAMPLE_PREPARE = 0,
  SAMPLE_SCORES = 1,
  SAMPLE_COARSE_FILTER = 2,
  SAMPLE_RADIX_0 = 3,
  SAMPLE_RADIX_1 = 4,
  SAMPLE_RADIX_2 = 5,
  SAMPLE_RADIX_3 = 6,
  SAMPLE_MASKED_FILTER = 7,
  SAMPLE_COLLECT = 8,
  SAMPLE_FINISH = 9,
};

enum ParallelSamplingRoute {
  PARALLEL_SAMPLE_REJECTION,
  PARALLEL_SAMPLE_RADIX,
};

struct ParallelSamplingState {
  // CTA completion counters for the coarse, score, and candidate phases.
  int arrivals[3];
  int radix_arrivals[4];
  // Request identity and output location.
  int active;
  int done;
  int row;
  int logit_row;
  int position;
  int vocab;
  int parts;
  // Top-k selection state; k becomes zero after the radix cutoff is known.
  int k;
  int best_token;
  int token;
  int selected_part;
  int remaining_k;
  int boundary_bucket;
  int radix_rank;
  int radix_done;
  uint32_t radix_prefix;
  uint32_t radix_mask;
  // Rejection sampler state shared by the finishing CTA.
  int draw_count;
  bool sample_boundary;
  bool rejected;
  ParallelSamplingRoute route;
  float maximum;
  float temperature;
  float candidate_weight;
  float candidate_score;
  float lower_weight;
  float lower_score;
  double p;
  double target;
  double residual;
  double tile_mass;
  double accepted[PARALLEL_SAMPLING_MAX_PARTS];
  double boundary_mass[PARALLEL_SAMPLING_MAX_PARTS];
};

struct ParallelSamplingPart {
  uint64_t best;
  Draw draw;
  uint64_t significands;
  int count, candidate_count;
  // The coarse histogram is no longer needed when radix selection starts.
  // Phase 7 rebuilds it after the final radix pass has completed.
  union {
    struct {
      uint32_t counts[2 * PARALLEL_SAMPLING_MASS_BINS];
      unsigned long long mantissas[2 * PARALLEL_SAMPLING_MASS_BINS];
    } histogram;
    uint32_t radix_counts[256];
  };
};

static_assert(sizeof(ParallelSamplingState) <=
              PARALLEL_SAMPLING_STATE_WORDS * sizeof(float));
static_assert(sizeof(ParallelSamplingPart) <=
              PARALLEL_SAMPLING_PART_WORDS * sizeof(float));
constexpr int PARALLEL_CUB_OFFSET = 12 * 1024;
constexpr int PARALLEL_SAMPLE_WIDTH = 16;

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

// Reuse the idle history count row for weights when frequency penalty is off.
// Otherwise keep the counts and recompute weights from the input logits.
template <typename T> struct ParallelSamplingReader {
  T const *logits;
  SamplingScoreReader score_reader;
  ParallelSamplingState const &state;
  float *cached_weights;
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
        cached_weights(cfg->frequency_penalty == 0 ? history_buffer : nullptr),
        plain_logits(cfg->bias_count == 0 && cfg->repetition_penalty == 1 &&
                     cfg->frequency_penalty == 0 && cfg->presence_penalty == 0) {}

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
    if (cached_weights) {
      return cached_weights[token];
    }
    float score = scaled_score(token);
    // After radix top-k, exclude scores below the selected cutoff.
    if (state.radix_done && score_bits(score) < state.radix_prefix) {
      return 0.f;
    }
    return expf(score);
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

__device__ inline double parallel_bucket_mass(ParallelSamplingPart const &part,
                                              int bucket) {
  double mass = 0;
  for (int sub = 0; sub < 2; ++sub) {
    int index = 2 * bucket + sub;
    uint32_t exponent = 2 * bucket + sub;
    double sig = double(part.histogram.mantissas[index]);
    if (exponent) {
      sig += double(part.histogram.counts[index]) * 8388608.0;
    }
    double unit =
        exponent ? double(__uint_as_float(exponent << 23)) * 0x1p-23 : 0x1p-149;
    mass += sig * unit;
  }
  return mass;
}

// All weights in the boundary bucket have one of two adjacent exponents.
// Accumulate their aligned significands as integers for reproducible sums.
__device__ inline uint64_t parallel_boundary_significand(float weight,
                                                         int bucket) {
  uint32_t bits = __float_as_uint(weight);
  uint32_t exponent = (bits >> 23) & 255;
  uint64_t sig = (bits & 0x7fffffu) + (exponent ? 0x800000u : 0u);
  return bucket ? sig << (exponent & 1) : sig;
}

__device__ inline double parallel_boundary_mass(uint64_t sig, int bucket) {
  uint32_t exponent = 2 * bucket;
  double unit =
      exponent ? double(__uint_as_float(exponent << 23)) * 0x1p-23 : 0x1p-149;
  return double(sig) * unit;
}

template <typename T>
__device__ inline bool
    parallel_above_candidate(ParallelSamplingScratch const &scratch,
                             ParallelSamplingReader<T> const &reader,
                             int token) {
  auto const &s = scratch.state;
  // Top-k compares scores; top-p compares their probability weights.
  return s.k ? reader.scaled_score(token) > s.candidate_score
             : reader.weight(token) > s.candidate_weight;
}

template <typename T>
__device__ inline bool
    parallel_eligible_boundary(ParallelSamplingScratch const &scratch,
                               ParallelSamplingReader<T> const &reader,
                               int token) {
  auto const &s = scratch.state;
  if (!s.rejected) {
    return true;
  }
  return s.k ? reader.scaled_score(token) > s.lower_score
             : reader.weight(token) > s.lower_weight;
}

__device__ inline void parallel_choose_part(ParallelSamplingState &s,
                                            ServingConfig const *cfg) {
  double higher_mass = 0, boundary_mass = 0;
  for (int j = 0; j < s.parts; ++j) {
    higher_mass += s.accepted[j];
    boundary_mass += s.boundary_mass[j];
  }
  SamplingRng rng{uint64_t(cfg->seed), uint32_t(s.position)};
  double target = (double(rng.bits(s.draw_count++)) + 0.5) * 0x1p-32 *
                  (higher_mass + boundary_mass);
  s.sample_boundary = target >= higher_mass;
  if (s.sample_boundary) {
    target -= higher_mass;
  }
  double const *masses = s.sample_boundary ? s.boundary_mass : s.accepted;
  int last = 0;
  for (int j = 0; j < s.parts; ++j) {
    if (masses[j] > 0) {
      last = j;
    }
    if (target < masses[j]) {
      s.selected_part = j;
      s.residual = target;
      return;
    }
    target -= masses[j];
  }
  s.selected_part = last;
  s.residual = nextafter(masses[last], 0.0);
}

template <int Threads, typename T>
__device__ inline void
    parallel_draw_partition(ParallelSamplingScratch const &scratch,
                            ParallelSamplingReader<T> const &reader,
                            bool use_candidates) {
  auto &s = scratch.state;
  using Scan = cub::BlockScan<double, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto &storage = *reinterpret_cast<typename Scan::TempStorage *>(
      sampling_shared + PARALLEL_CUB_OFFSET);
  int begin = s.vocab * s.selected_part / s.parts;
  int end = s.vocab * (s.selected_part + 1) / s.parts;
  int count = use_candidates && s.sample_boundary
                  ? scratch.part(s.selected_part).candidate_count
                  : end - begin;
  double previous = 0;
  int last = -1;
  if (threadIdx.x == 0) {
    s.token = INT_MAX;
  }
  __syncthreads();
  for (int base = 0; base < count; base += Threads * PARALLEL_SAMPLE_WIDTH) {
    double weights[PARALLEL_SAMPLE_WIDTH], cumulative[PARALLEL_SAMPLE_WIDTH];
    for (int i = 0; i < PARALLEL_SAMPLE_WIDTH; ++i) {
      int index = base + int(threadIdx.x) * PARALLEL_SAMPLE_WIDTH + i;
      int token = index < count && use_candidates && s.sample_boundary
                      ? scratch.candidate_token(begin, index)
                      : begin + index;
      float weight = index < count ? reader.weight(token) : 0;
      int bucket = __float_as_uint(weight) >> 24;
      bool eligible = s.sample_boundary
                          ? bucket == s.boundary_bucket &&
                                (index >= count ||
                                 parallel_eligible_boundary(scratch, reader, token))
                          : bucket > s.boundary_bucket;
      weights[i] = index < count && eligible ? weight : 0;
      if (weights[i] > 0) {
        last = token;
      }
    }
    Scan(storage).InclusiveSum(weights, cumulative);
    if (threadIdx.x == Threads - 1) {
      s.tile_mass = cumulative[PARALLEL_SAMPLE_WIDTH - 1];
    }
    __syncthreads();
    for (int i = 0; i < PARALLEL_SAMPLE_WIDTH; ++i) {
      if (weights[i] > 0 && previous + cumulative[i] > s.residual &&
          previous + cumulative[i] - weights[i] <= s.residual) {
        int index = base + int(threadIdx.x) * PARALLEL_SAMPLE_WIDTH + i;
        int token = use_candidates && s.sample_boundary
                        ? scratch.candidate_token(begin, index)
                        : begin + index;
        atomicMin(&s.token, token);
      }
    }
    __syncthreads();
    if (s.token != INT_MAX) {
      break;
    }
    previous += s.tile_mass;
    __syncthreads();
  }
  last = cub_block_reduce<Threads>(last, MaximumIndex{});
  if (threadIdx.x == 0) {
    if (s.token == INT_MAX) {
      s.token = last >= 0 ? last : s.best_token;
    }
    s.candidate_weight = reader.weight(s.token);
    s.candidate_score = reader.scaled_score(s.token);
    if (!s.sample_boundary) {
      s.done = true;
    }
  }
  __syncthreads();
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
      s.remaining_k = s.k;
      s.temperature = fmaxf(float(cfg->temperature), 0x1p-126f);
      s.p = cfg->top_p;
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

// Phase 1: process logits, find the row maximum, and handle greedy/unfiltered
// draws.
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
  int begin = s.vocab * partition / s.parts,
      end = s.vocab * (partition + 1) / s.parts;
  uint64_t best = 0;
  Draw draw;
  ChooseDraw choose{s.temperature};
  SamplingRng rng{uint64_t(cfg->seed), uint32_t(s.position)};
  scan_logits<4>(logits + begin, end - begin, [&](int offset, float logit) {
    int v = begin + offset;
    float score = reader.processed_score(v, logit);
    uint64_t key = score_key(score, v);
    best = key > best ? key : best;
    if (unfiltered) {
      draw =
          choose(draw, Draw{score, isfinite(score) ? rng.gumbel(v) : 0.f, v});
    }
  });
  best = cub_block_reduce<Threads>(best, Maximum{});
  if (unfiltered) {
    draw = cub_block_reduce<Threads>(draw, choose);
  }
  if (threadIdx.x == 0) {
    scratch.part(partition).best = best;
    scratch.part(partition).draw = draw;
  }
  if (!parallel_last_arrival(s, 1)) {
    return;
  }
  if (threadIdx.x == 0) {
    best = 0;
    Draw winner;
    for (int j = 0; j < s.parts; ++j) {
      auto &part = scratch.part(j);
      if (part.best > best) {
        best = part.best;
      }
      if (unfiltered) {
        winner = choose(winner, part.draw);
      }
    }
    s.best_token = 0xffffffffu - uint32_t(best);
    s.maximum = reader.processed_score(s.best_token);
    if (greedy || !isfinite(s.maximum) || unfiltered) {
      s.token = unfiltered ? winner.token : s.best_token;
      s.done = true;
    }
  }
}

// Phases 2 and 7: build a probability histogram. Phase 7 masks logits below
// the top-k cutoff before applying top-p to that retained set.
template <int Threads, typename T>
__device__ inline void
    parallel_coarse_filter(ParallelSamplingScratch const &scratch,
                           ParallelSamplingReader<T> const &reader,
                           ServingConfig const *cfg,
                           int partition,
                           bool masked = false) {
  auto &s = scratch.state;
  if (!s.active || s.done || (masked && s.route != PARALLEL_SAMPLE_RADIX)) {
    return;
  }
  constexpr int Bins = PARALLEL_SAMPLING_MASS_BINS;
  constexpr int Slots = 2 * Bins;
  constexpr int Warps = Threads / 32;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto *low = reinterpret_cast<uint32_t *>(sampling_shared);
  auto *high = low + Slots * Warps;
  auto *counts = high + Slots * Warps;
  // Each warp accumulates its own histogram to reduce shared-memory contention.
  static_assert(Slots * Warps * 12 <= PARALLEL_CUB_OFFSET);
  for (int i = threadIdx.x; i < Slots * Warps; i += Threads) {
    low[i] = high[i] = counts[i] = 0;
  }
  __syncthreads();
  int begin = s.vocab * partition / s.parts,
      end = s.vocab * (partition + 1) / s.parts;
  int warp_offset = (threadIdx.x / 32) * Slots;
  float maximum = s.maximum, temperature = s.temperature;
  scan_logits<1>(reader.logits + begin, end - begin, [&](int offset, float logit) {
    int token = begin + offset;
    float score = reader.processed_score(token, logit);
    float scaled_score = (score - maximum) / temperature;
    float weight = expf(scaled_score);
    if (masked && score_bits(scaled_score) < s.radix_prefix) {
      weight = 0;
    }
    if (reader.cached_weights) {
      reader.cached_weights[token] = weight;
    }
    uint32_t key = __float_as_uint(weight);
    uint32_t slot = 2 * (key >> 24) + ((key >> 23) & 1);
    uint32_t sig = key & 0x7fffffu;
    // Shared 64-bit atomicAdd is a CAS loop on Ampere. Use native
    // 32-bit additions and propagate the carry to keep the sum exact.
    int index = warp_offset + slot;
    uint32_t previous = atomicAdd(low + index, sig);
    if (previous > UINT_MAX - sig) {
      atomicAdd(high + index, 1u);
    }
    atomicAdd(counts + warp_offset + slot, 1u);
  });
  __syncthreads();
  for (int i = threadIdx.x; i < Slots; i += Threads) {
    uint32_t count = 0;
    unsigned long long sum = 0;
    for (int w = 0; w < Warps; ++w) {
      count += counts[w * Slots + i];
      int index = w * Slots + i;
      sum += (uint64_t(high[index]) << 32) | low[index];
    }
    scratch.part(partition).histogram.counts[i] = count;
    scratch.part(partition).histogram.mantissas[i] = sum;
  }
  if (!parallel_last_arrival(s, 0)) {
    return;
  }
  // The last CTA combines partition histograms and finds the top-k or top-p
  // boundary bucket. The other CTAs are finished with this phase.
  int bucket = Bins - 1 - int(threadIdx.x);
  int count = 0;
  double mass = 0;
  if (bucket >= 0) {
    // Only these two histogram entries are consumed below. Keep aggregates
    // in registers instead of constructing another shared histogram.
    unsigned long long sums[2] = {0, 0};
    uint32_t numbers[2] = {0, 0};
    for (int j = 0; j < s.parts; ++j) {
      for (int sub = 0; sub < 2; ++sub) {
        sums[sub] += scratch.part(j).histogram.mantissas[2 * bucket + sub];
        numbers[sub] += scratch.part(j).histogram.counts[2 * bucket + sub];
      }
    }
    for (int sub = 0; sub < 2; ++sub) {
      uint32_t exponent = 2 * bucket + sub;
      double sig =
          double(sums[sub]) + (exponent ? double(numbers[sub]) * 8388608.0 : 0);
      double unit = exponent ? double(__uint_as_float(exponent << 23)) * 0x1p-23
                             : 0x1p-149;
      count += numbers[sub];
      mass += sig * unit;
    }
  }
  using CountScan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  using MassScan = cub::BlockScan<double, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  auto &count_storage = *reinterpret_cast<typename CountScan::TempStorage *>(
      sampling_shared + PARALLEL_CUB_OFFSET);
  auto &mass_storage = *reinterpret_cast<typename MassScan::TempStorage *>(
      sampling_shared + PARALLEL_CUB_OFFSET);
  int cumulative_count;
  double cumulative_mass;
  CountScan(count_storage).InclusiveSum(count, cumulative_count);
  __syncthreads();
  MassScan(mass_storage).InclusiveSum(mass, cumulative_mass);
  if (threadIdx.x == Threads - 1) {
    s.target = s.p * cumulative_mass;
  }
  __syncthreads();
  bool selected =
      s.k ? cumulative_count >= s.k && cumulative_count - count < s.k
          : mass > 0 && cumulative_mass >= s.target &&
                cumulative_mass - mass < s.target;
  if (selected) {
    s.boundary_bucket = bucket;
    s.remaining_k = s.k ? s.k - (cumulative_count - count) : 0;
    s.residual = cumulative_mass - mass;
  }
  __syncthreads();
  // Split probability mass above and inside the boundary by partition. These
  // totals let the finishing CTA choose a partition before scanning tokens.
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  for (int j = warp; j < s.parts; j += Warps) {
    double higher = 0;
    for (int b = s.boundary_bucket + 1 + lane; b < Bins; b += 32) {
      higher += parallel_bucket_mass(scratch.part(j), b);
    }
    for (int delta = 16; delta; delta /= 2) {
      higher += __shfl_down_sync(0xffffffffu, higher, delta);
    }
    if (lane == 0) {
      s.accepted[j] = higher;
      s.boundary_mass[j] =
          parallel_bucket_mass(scratch.part(j), s.boundary_bucket);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    s.target -= s.residual;
    s.route =
        s.k && s.p < 1 ? PARALLEL_SAMPLE_RADIX : PARALLEL_SAMPLE_REJECTION;
    if (s.route == PARALLEL_SAMPLE_RADIX) {
      s.radix_rank = s.remaining_k;
      s.radix_done = 0;
      s.radix_prefix = 0;
      s.radix_mask = 0;
      s.arrivals[0] = 0;
    }
    if (s.route == PARALLEL_SAMPLE_REJECTION) {
      parallel_choose_part(s, cfg);
    }
  }
  __syncthreads();
  // H is wholly inside the retained set; only the boundary bucket B is
  // unresolved. Sampling H+B and rejecting only proven outsiders preserves
  // the target distribution without completing every radix digit.
  if (s.route == PARALLEL_SAMPLE_REJECTION) {
    parallel_draw_partition<Threads>(scratch, reader, false);
  }
}

// Phases 3-6: resolve the top-k boundary by one score byte per phase.
// Only requests combining top-k and top-p enter this path.
template <int Threads, int Round, typename T>
__device__ inline void
    parallel_radix_top_k(ParallelSamplingScratch const &scratch,
                         ParallelSamplingReader<T> const &reader,
                         int partition) {
  auto &s = scratch.state;
  if (!s.active || s.done || s.radix_done || s.route != PARALLEL_SAMPLE_RADIX) {
    return;
  }
  constexpr int Bins = 256;
  constexpr int Warps = Threads / 32;
  constexpr int Shift = 24 - 8 * Round;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto *histogram = reinterpret_cast<unsigned int *>(sampling_shared);
  for (int i = threadIdx.x; i < Bins * Warps; i += Threads) {
    histogram[i] = 0;
  }
  __syncthreads();
  int begin = s.vocab * partition / s.parts;
  int end = s.vocab * (partition + 1) / s.parts;
  uint32_t prefix = s.radix_prefix, mask = s.radix_mask;
  int warp_offset = threadIdx.x / 32 * Bins;
  auto &part = scratch.part(partition);
  if constexpr (Round == 0) {
    if (threadIdx.x == 0) {
      part.candidate_count = 0;
    }
  }
  __syncthreads();
  int count = Round == 0 ? end - begin : part.candidate_count;
  // The first pass visits the whole partition and compacts the coarse
  // boundary bucket. Later passes only revisit those candidate token IDs.
  for (int index = threadIdx.x; index < count; index += Threads) {
    int token = Round == 0 ? begin + index
                           : scratch.candidate_token(begin, index);
    if constexpr (Round == 0) {
      int bucket = __float_as_uint(reader.weight(token)) >> 24;
      if (bucket != s.boundary_bucket) {
        continue;
      }
      int offset = atomicAdd(&part.candidate_count, 1);
      scratch.store_candidate(begin, offset, token);
    }
    float normalized = reader.scaled_score(token);
    uint32_t bits = score_bits(normalized);
    if ((bits & mask) == prefix) {
      atomicAdd(histogram + warp_offset + ((bits >> Shift) & 255), 1u);
    }
  }
  __syncthreads();
  auto *counts = part.radix_counts;
  for (int i = threadIdx.x; i < Bins; i += Threads) {
    int count = 0;
    for (int warp = 0; warp < Warps; ++warp) {
      count += histogram[warp * Bins + i];
    }
    counts[i] = count;
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
  // One CTA merges all radix histograms and descends into the bin containing
  // the remaining top-k rank. Extra threads contribute zero when Threads=256.
  using Scan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  auto &storage = *reinterpret_cast<typename Scan::TempStorage *>(
      sampling_shared + PARALLEL_CUB_OFFSET);
  static_assert(PARALLEL_CUB_OFFSET + sizeof(typename Scan::TempStorage) <=
                PARALLEL_SAMPLING_SHARED_BYTES);
  int bin_counts[2], cumulative[2];
  for (int i = 0; i < 2; ++i) {
    int bin = Bins - 1 - int(threadIdx.x) * 2 - i;
    int count = 0;
    if (bin >= 0) {
      for (int part = 0; part < s.parts; ++part) {
        count += scratch.part(part).radix_counts[bin];
      }
    }
    bin_counts[i] = count;
  }
  Scan(storage).InclusiveSum(bin_counts, cumulative);
  int rank = s.radix_rank;
  for (int i = 0; i < 2; ++i) {
    int bin = Bins - 1 - int(threadIdx.x) * 2 - i;
    if (bin >= 0 && cumulative[i] >= rank &&
        cumulative[i] - bin_counts[i] < rank) {
      s.radix_prefix |= uint32_t(bin) << Shift;
      s.radix_mask |= 255u << Shift;
      int remaining = rank - (cumulative[i] - bin_counts[i]);
      s.radix_rank = remaining;
      if (remaining == bin_counts[i] || Round == 3) {
        // The whole selected bucket is retained, including boundary ties.
        s.radix_done = 1;
        s.k = 0;
      }
    }
  }
}

// Shared acceptance test for the first boundary draw and later refinements.
__device__ inline void
    parallel_test_candidate(ParallelSamplingScratch const &scratch) {
  auto &s = scratch.state;
  uint64_t sig = 0;
  int count = 0;
  for (int j = 0; j < s.parts; ++j) {
    sig += scratch.part(j).significands;
    count += scratch.part(j).count;
  }
  bool accepted =
      s.k ? count < s.remaining_k
          : parallel_boundary_mass(sig, s.boundary_bucket) < s.target;
  if (accepted) {
    s.done = true;
  } else {
    s.rejected = true;
    s.lower_weight = s.candidate_weight;
    s.lower_score = s.candidate_score;
    for (int j = 0; j < s.parts; ++j) {
      s.boundary_mass[j] = parallel_boundary_mass(scratch.part(j).significands,
                                                  s.boundary_bucket);
    }
  }
}

// Phase 8: collect the unresolved boundary bucket in token order.
template <int Threads, typename T>
__device__ __noinline__ void
    parallel_collect(ParallelSamplingScratch const &scratch,
                     ParallelSamplingReader<T> const &reader,
                     int partition) {
  auto &s = scratch.state;
  if (!s.active || s.done || s.route == PARALLEL_SAMPLE_RADIX) {
    return;
  }
  auto &part = scratch.part(partition);
  using Scan = cub::BlockScan<int, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  extern __shared__ __align__(16) unsigned char sampling_shared[];
  auto &storage = *reinterpret_cast<typename Scan::TempStorage *>(
      sampling_shared + PARALLEL_CUB_OFFSET);
  int begin = s.vocab * partition / s.parts,
      end = s.vocab * (partition + 1) / s.parts;
  uint64_t sig = 0;
  int offset = 0;
  // Stable compaction makes CDF traversal independent of warp arrival order.
  for (int base = begin; base < end; base += Threads * 4) {
    unsigned selected = 0;
    for (int i = 0; i < 4; ++i) {
      int token = base + int(threadIdx.x) * 4 + i;
      float weight = token < end ? reader.weight(token) : 0;
      int bucket = __float_as_uint(weight) >> 24;
      bool eligible = weight > 0 && bucket == s.boundary_bucket &&
                      token < end && parallel_above_candidate(scratch, reader, token);
      if (token < end && eligible) {
        selected |= 1u << i;
      }
    }
    int prefix, total;
    Scan(storage).ExclusiveSum(__popc(selected), prefix, total);
    for (int i = 0; i < 4; ++i) {
      if (selected & (1u << i)) {
        int token = base + int(threadIdx.x) * 4 + i;
        int index = offset + prefix++;
        scratch.store_candidate(begin, index, token);
        sig += parallel_boundary_significand(reader.weight(token),
                                             s.boundary_bucket);
      }
    }
    offset += total;
    __syncthreads();
  }
  sig = cub_block_reduce<Threads>(sig, cuda::std::plus<uint64_t>{});
  if (threadIdx.x == 0) {
    part.candidate_count = offset;
    part.significands = sig;
    part.count = offset;
  }
  if (parallel_last_arrival(s, 2) && threadIdx.x == 0) {
    parallel_test_candidate(scratch);
  }
}

template <int Threads, typename T>
__device__ inline void parallel_refine(ParallelSamplingScratch const &scratch,
                                      ParallelSamplingReader<T> const &reader) {
  auto &s = scratch.state;
  int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  for (int j = warp; j < s.parts; j += Threads / 32) {
    auto &part = scratch.part(j);
    int begin = s.vocab * j / s.parts;
    uint64_t sig = 0;
    int count = 0;
    for (int i = lane; i < part.candidate_count; i += 32) {
      int token = scratch.candidate_token(begin, i);
      if (parallel_above_candidate(scratch, reader, token)) {
        sig += parallel_boundary_significand(reader.weight(token),
                                             s.boundary_bucket);
        ++count;
      }
    }
    for (int delta = 16; delta; delta /= 2) {
      sig += __shfl_down_sync(0xffffffffu, sig, delta);
      count += __shfl_down_sync(0xffffffffu, count, delta);
    }
    if (lane == 0) {
      part.significands = sig;
      part.count = count;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    parallel_test_candidate(scratch);
  }
  __syncthreads();
}

// Phase 9: draw from the retained set and refine the boundary after rejection.
template <int Threads, typename T>
__device__ __noinline__ void
    parallel_finish(ParallelSamplingScratch const &scratch,
                    ParallelSamplingReader<T> const &reader,
                    long long *output,
                    ServingConfig const *cfg) {
  auto &s = scratch.state;
  if (!s.active) {
    return;
  }
  while (!s.done) {
    if (threadIdx.x == 0) {
      parallel_choose_part(s, cfg);
    }
    __syncthreads();
    parallel_draw_partition<Threads>(scratch, reader, true);
    if (!s.done) {
      parallel_refine<Threads>(scratch, reader);
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
      padded_vocab, parts);
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
    if (!s.active) {
      return;
    }
    auto *cfg = config.generation_config + s.row;
    auto *row_logits = logits + s.logit_row * padded_vocab;
    auto *row_history = history_buffer +
                        s.row * sampling_scratch_words(padded_vocab);
    if constexpr (Phase == SAMPLE_SCORES) {
      parallel_scores<Threads>(row_logits,
                               row_history,
                               scratch,
                               cfg,
                               partition,
                               padded_vocab);
    } else {
      ParallelSamplingReader reader(row_logits, row_history, padded_vocab, cfg,
                                    s);
      if constexpr (Phase == SAMPLE_COARSE_FILTER) {
        parallel_coarse_filter<Threads>(scratch, reader, cfg, partition);
      } else if constexpr (Phase >= SAMPLE_RADIX_0 && Phase <= SAMPLE_RADIX_3) {
        parallel_radix_top_k<Threads, Phase - SAMPLE_RADIX_0>(scratch, reader,
                                                              partition);
      } else if constexpr (Phase == SAMPLE_MASKED_FILTER) {
        parallel_coarse_filter<Threads>(scratch, reader, cfg, partition, true);
      } else if constexpr (Phase == SAMPLE_COLLECT) {
        parallel_collect<Threads>(scratch, reader, partition);
      } else if constexpr (Phase == SAMPLE_FINISH) {
        parallel_finish<Threads>(scratch, reader, output, cfg);
      }
    }
  }
}

#endif
} // namespace serving
} // namespace mirage
