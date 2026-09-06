#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

namespace mirage { namespace serving {
constexpr int CONFIG_WORDS = 544;

__device__ inline float option(int64_t value) {
  return static_cast<float>(__longlong_as_double(value));
}
__device__ inline uint64_t score_key(float score, uint32_t token) {
  uint32_t bits = __float_as_uint(score == 0.f ? 0.f : score);
  bits = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (uint64_t(bits) << 32) | (0xffffffffu - token);
}
__device__ inline uint64_t mix(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
// One CTA owns an output token. Reductions work on Ampere (128 threads) and
// Hopper/Blackwell (256 threads), without architecture-specific dependencies.
__device__ inline float sum(float value, float *shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  float result = shared[0];
  __syncthreads();
  return result;
}
__device__ inline uint64_t maximum(uint64_t value, uint64_t *shared) {
  shared[threadIdx.x] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride && shared[threadIdx.x + stride] > shared[threadIdx.x])
      shared[threadIdx.x] = shared[threadIdx.x + stride];
    __syncthreads();
  }
  uint64_t result = shared[0];
  __syncthreads();
  return result;
}

// Exact radix selection with deterministic token-ID tie breaking. Counting
// selects top-k; summing probability mass selects the smallest nucleus.
// Complexity is O(64 * vocabulary), independent of k and nucleus size.
__device__ inline uint64_t cutoff(float *scores, int vocab, uint64_t lower,
                                  float target, bool weighted, float max_score,
                                  float *shared) {
  uint64_t prefix = 0, mask = 0;
  for (int bit = 63; bit >= 0; --bit) {
    uint64_t test = uint64_t(1) << bit;
    float mass = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
      uint64_t key = score_key(scores[v], v);
      if (key >= lower && (key & mask) == prefix && (key & test))
        mass += weighted ? expf(scores[v] - max_score) : 1.f;
    }
    mass = sum(mass, shared);
    mask |= test;
    if (mass > 0 && mass >= target) prefix |= test;
    else target -= mass;
  }
  return prefix;
}

template<typename T, typename Token>
__device__ inline void sample(T const *logits, float *scratch, Token *output,
                              int padded_vocab, int64_t const *cfg,
                              long long const *history, int history_len,
                              int prompt_len, int generation_position) {
  __shared__ float reduction[256];
  __shared__ uint64_t keys[256];
  int vocab = min(padded_vocab, int(cfg[2]));
  float *scores = scratch;
  int *counts = reinterpret_cast<int *>(scratch + padded_vocab);
  int *generated_counts = counts + padded_vocab;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    counts[v] = 0;
    generated_counts[v] = 0;
  }
  __syncthreads();
  bool penalties = option(cfg[6]) != 0 || option(cfg[7]) != 0 || option(cfg[8]) != 1;
  if (penalties) {
    for (int i = threadIdx.x; i < history_len; i += blockDim.x) {
      int64_t token = history[i];
      if (token >= 0 && token < vocab) {
        atomicAdd(counts + token, 1);
        if (i >= prompt_len) atomicAdd(generated_counts + token, 1);
      }
    }
  }
  __syncthreads();
  float temperature = option(cfg[3]);
  uint64_t best = 0;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    float score = static_cast<float>(logits[v]);
    for (int j = 0; j < cfg[10]; ++j)
      if (cfg[32 + 2*j] == v) score += option(cfg[33 + 2*j]);
    if (counts[v]) score = score > 0 ? score / option(cfg[8]) : score * option(cfg[8]);
    score -= option(cfg[6]) * generated_counts[v] + option(cfg[7]) * (generated_counts[v] > 0);
    if (temperature > 0) score /= temperature;
    // NaNs never win. Actual -inf logits remain masked.
    if (isnan(score)) score = -INFINITY;
    scores[v] = score;
    uint64_t key = score_key(score, v);
    best = key > best ? key : best;
  }
  best = maximum(best, keys);
  int best_token = 0xffffffffu - uint32_t(best);
  if (temperature == 0) {
    if (threadIdx.x == 0) *output = best_token;
    return;
  }
  float max_score = scores[best_token];
  if (!isfinite(max_score)) {
    if (threadIdx.x == 0) *output = best_token;
    return;
  }
  uint64_t lower = 0;
  int k = int(cfg[5]);
  if (k > 0 && k < vocab)
    lower = cutoff(scores, vocab, 0, float(k), false, max_score, reduction);
  float top_p = option(cfg[4]);
  if (top_p < 1.f) {
    float mass = 0;
    for (int v = threadIdx.x; v < vocab; v += blockDim.x)
      if (score_key(scores[v], v) >= lower) mass += expf(scores[v] - max_score);
    mass = sum(mass, reduction);
    lower = cutoff(scores, vocab, lower, top_p * mass, true, max_score, reduction);
  }
  best = 0;
  for (int v = threadIdx.x; v < vocab; v += blockDim.x) {
    if (score_key(scores[v], v) < lower || !isfinite(scores[v])) continue;
    uint64_t random = mix(uint64_t(cfg[1]) ^ mix(uint64_t(generation_position)) ^ mix(uint64_t(v) + 1));
    double u = (double(random >> 11) + 0.5) * 0x1.0p-53;
    float noisy = scores[v] - max_score - float(log(-log(u)));
    uint64_t key = score_key(noisy, v);
    best = key > best ? key : best;
  }
  best = maximum(best, keys);
  if (threadIdx.x == 0) *output = 0xffffffffu - uint32_t(best);
}

#if defined(MODE_ONLINE_PINNED)
template<typename T>
__device__ inline void sample_batch(T const *logits, float *scratch, long long *output,
                                    int padded_vocab, mirage::runtime::RuntimeConfig const &config) {
  // This task sees the whole logits tensor. Only sample the last prefill
  // position or a decode position; discarded prefill logits consume no RNG.
  for (int slot = 0; slot < MPK_MAX_NUM_BATCHED_REQUESTS; ++slot) {
    int row = config.request_ids[slot];
    if (row < 0) continue;
    int start = config.qo_indptr_buffer[slot], end = config.qo_indptr_buffer[slot + 1];
    for (int i = start; i < end; ++i) {
      int position = config.step[row] + i - start;
      int prompt = config.prompt_length[row];
      if (position + 1 < prompt) continue;
      sample(logits + i * padded_vocab, scratch, output + i, padded_vocab,
             config.generation_config + row * CONFIG_WORDS,
             config.tokens + row * MPK_MAX_SEQ_LENGTH, position + 1, prompt,
             position + 1 - prompt);
      __syncthreads();
    }
  }
}
#endif
}} // namespace mirage::serving
