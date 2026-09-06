"""Compile and exercise the actual serving sampler, without the graph compiler.
Run MIRAGE_TEST_GPUS=0,1,2,3 python -m pytest tests/engine/test_sampling_gpu.py.
"""
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from mirage.engine.protocol import SamplingParams

GPUS = [int(x) for x in os.environ.get("MIRAGE_TEST_GPUS", "").split(",") if x]
pytestmark = pytest.mark.skipif(not GPUS, reason="set MIRAGE_TEST_GPUS for GPU tests")


@pytest.fixture(scope="module")
def sampler(tmp_path_factory):
    output = tmp_path_factory.mktemp("cuda") / "sampling.so"
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["/usr/local/cuda/bin/nvcc", "-std=c++17", "-arch=sm_86", "-shared",
                    "-Xcompiler=-fPIC", "-I" + str(root / "include"),
                    str(Path(__file__).with_name("sampling_cuda.cu")), "-o", str(output)], check=True)
    lib = ctypes.CDLL(str(output))
    lib.sample_test.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_void_p]
    def run(device, logits, params, history=(), prompt_len=0, position=0):
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        if logits.ndim == 1:
            logits = np.tile(logits, (len(params), 1))
        configs = np.array([p.pack(1, 100, logits.shape[1], []) for p in params], dtype=np.int64)
        history = np.array(history, dtype=np.int64)
        out = np.zeros(len(params), dtype=np.int64)
        err = lib.sample_test(device, logits.ctypes.data, configs.ctypes.data, history.ctypes.data,
                              len(history), prompt_len, position, logits.shape[1], len(params), out.ctypes.data)
        assert err == 0, f"CUDA error {err}"
        return out
    return run


@pytest.mark.parametrize("device", GPUS or [0])
def test_sampling_distribution(sampler, device):
    logits = np.array([-.8, .2, 1.2, 1.7, -.2, -1.5, 0, .7])
    for temperature, top_k, top_p in [(1., 0, 1.), (.5, 0, 1.), (1., 3, 1.), (.8, 0, .7), (1., 4, .8)]:
        params = [SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, seed=i) for i in range(12000)]
        actual = sampler(device, logits, params)
        scores = logits / temperature
        order = np.argsort(-scores, kind="stable")
        if top_k:
            order = order[:top_k]
        probs = np.exp(scores[order] - scores[order].max())
        probs /= probs.sum()
        count = np.searchsorted(np.cumsum(probs), top_p) + 1
        order, probs = order[:count], probs[:count]
        probs /= probs.sum()
        expected = np.zeros(len(logits)); expected[order] = probs
        empirical = np.bincount(actual, minlength=len(logits)) / len(actual)
        assert np.max(np.abs(empirical - expected)) < .025
        assert set(actual) <= set(order)


@pytest.mark.parametrize("device", GPUS or [0])
def test_greedy_penalties_and_seed(sampler, device):
    assert sampler(device, [1., 2., 3.], [SamplingParams()])[0] == 2
    assert sampler(device, [1., 2., 3.], [SamplingParams(logit_bias={0: 10})])[0] == 0
    assert sampler(device, [1., 2., 3.], [SamplingParams(repetition_penalty=2)], [2], 1)[0] == 1
    assert sampler(device, [1., 2., 3.], [SamplingParams(frequency_penalty=2)], [2], 0)[0] == 1
    assert sampler(device, [1., 2., 3.], [SamplingParams(presence_penalty=2)], [2], 0)[0] == 1
    # Prompt tokens do not contribute to frequency/presence penalties.
    assert sampler(device, [1., 2., 3.], [SamplingParams(frequency_penalty=2)], [2], 1)[0] == 2
    params = [SamplingParams(temperature=1, seed=i) for i in range(30)]
    first = sampler(device, [1., 2., 3.], params, position=9)
    shuffled = sampler(device, [1., 2., 3.], params[::-1], position=9)
    np.testing.assert_array_equal(first, shuffled[::-1])
    assert (first != sampler(device, [1., 2., 3.], params, position=10)).any()
    # Equal-score top-k ties prefer the lower token ID.
    assert set(sampler(device, [1., 1., 1.], [SamplingParams(temperature=1, top_k=1)]*10)) == {0}
