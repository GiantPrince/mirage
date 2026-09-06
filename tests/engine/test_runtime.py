"""CPU ring-publication and cancellation ownership tests."""
import collections
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("runtime_under_test", Path(__file__).resolve().parents[2] /
                                             "python/mirage/mpk/online_pinned_runtime.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Runtime = module.OnlinePinnedRuntime


@pytest.fixture
def runtime(monkeypatch):
    r = Runtime.__new__(Runtime)
    r._cap = 2; r._mask = 1; r._cpu_req_tail = 0
    r._ring_lock = threading.Lock(); r._waiting_lock = threading.Lock(); r._lock = threading.RLock()
    r._waiting = collections.deque(); r._completions = {}; r._cancelled = set(); r._abandoned = set()
    r._drain_error = None
    r._req_ready = torch.zeros(2, dtype=torch.int32)
    r._req_request_id = torch.zeros(2, dtype=torch.int32)
    r._req_prompt_len = torch.zeros(2, dtype=torch.int32)
    r._req_initial_step = torch.zeros(2, dtype=torch.int32)
    r._inbox_tokens = torch.zeros(2, 10, dtype=torch.int64)
    r._generation_config = torch.zeros(2, 544, dtype=torch.int64)
    r._pinned_rid_at_row = torch.tensor([10, -1], dtype=torch.int32)
    r._pinned_step = torch.zeros(2, dtype=torch.int32)
    r._cancel = torch.full((2,), -1, dtype=torch.int32)
    r._total_inflight = 2
    r._write_stream = SimpleNamespace(synchronize=lambda: None)
    r._load_i32_acquire = lambda tensor, i: int(tensor[i])
    def store(tensor, i, value):
        if tensor is r._req_ready and value == 1:
            assert r._generation_config[i, 1] == r._req_request_id[i] + 100
        tensor[i] = value
    r._store_i32_release = store
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    return r


def config(rid):
    result = [0] * 544
    result[1] = rid + 100
    return result


def test_ring_waiting_and_settings_are_copied(runtime):
    tokens = torch.tensor([1, 2])
    assert runtime.submit(1, tokens, generation_config=config(1))
    assert runtime.submit(2, tokens, generation_config=config(2))
    cfg = config(3)
    assert not runtime.submit(3, tokens, generation_config=cfg)
    tokens.fill_(9); cfg[1] = 0
    runtime._req_ready[0] = 0
    assert runtime.flush_waiting() == 1
    assert runtime._generation_config[0, 1] == 103
    assert runtime._inbox_tokens[0, :2].tolist() == [1, 2]


def test_cancel_waiting_and_row_identity(runtime):
    runtime._waiting.append((11, torch.tensor([1]), 0, torch.tensor(config(11))))
    runtime.abandon_request(11)
    assert not runtime._waiting
    assert 11 not in runtime._abandoned
    runtime.abandon_request(10)
    assert runtime._cancel.tolist() == [10, -1]
    assert 10 in runtime._abandoned
    # Cancellation records the old rid, so the signal cannot cancel a new lease.
    runtime._pinned_rid_at_row[0] = 12
    assert runtime._cancel[0] != runtime._pinned_rid_at_row[0]
    with pytest.raises(RuntimeError, match="belongs"):
        runtime._release_row_locked(10, 0)


def test_length_and_id_validation(runtime):
    for rid, tokens, step in [(2**31, torch.tensor([1]), 0), (1, torch.tensor([]), 0),
                               (1, torch.ones(10, dtype=torch.int64), 0), (1, torch.tensor([1]), 1)]:
        with pytest.raises(ValueError): runtime.submit(rid, tokens, step, config(1))
