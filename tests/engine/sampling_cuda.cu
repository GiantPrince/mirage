#include "mirage/persistent_kernel/tasks/common/serving_sampling.cuh"
#include <vector>

__global__ void run_sample(float const *logits, float *scratch, long long *out,
                           int vocab, int64_t const *configs, long long const *history,
                           int history_len, int prompt_len, int position) {
  int b = blockIdx.x;
  int scratch_stride = vocab * 3 + 768 + (vocab & 1);
  mirage::serving::sample(logits + b * vocab, scratch + b * scratch_stride, out + b,
                          vocab, configs + b * 544, history, history_len, prompt_len, position);
}
extern "C" int sample_test(int device, float const *logits, int64_t const *configs,
                            long long const *history, int history_len, int prompt_len,
                            int position, int vocab, int batch, long long *output) {
  cudaError_t err = cudaSetDevice(device);
  if (err != cudaSuccess) return int(err);
  float *dl = nullptr, *scratch = nullptr;
  int64_t *dc = nullptr;
  long long *dh = nullptr, *out = nullptr;
  auto cleanup = [&]() { cudaFree(dl); cudaFree(scratch); cudaFree(dc); cudaFree(dh); cudaFree(out); };
#define CHECK(call) do { err = call; if (err != cudaSuccess) { cleanup(); return int(err); } } while (0)
  CHECK(cudaMalloc(&dl, batch * vocab * sizeof(float)));
  CHECK(cudaMalloc(&scratch, batch * (vocab * 3 + 768 + (vocab & 1)) * sizeof(float)));
  CHECK(cudaMalloc(&dc, batch * 544 * sizeof(int64_t)));
  CHECK(cudaMalloc(&dh, (history_len + 1) * sizeof(long long)));
  CHECK(cudaMalloc(&out, batch * sizeof(long long)));
  CHECK(cudaMemcpy(dl, logits, batch * vocab * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dc, configs, batch * 544 * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dh, history, history_len * sizeof(long long), cudaMemcpyHostToDevice));
  run_sample<<<batch, 128>>>(dl, scratch, out, vocab, dc, dh, history_len, prompt_len, position);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(output, out, batch * sizeof(long long), cudaMemcpyDeviceToHost));
  cleanup();
  return 0;
}
