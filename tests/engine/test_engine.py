import threading
from types import SimpleNamespace

import pytest
import torch
from mirage.engine.llm_engine import LLMEngine
from mirage.engine.protocol import SamplingParams


class Tokenizer:
    chat_template = "test"
    def encode(self, text, **kwargs):
        return [1, 2]
    def decode(self, tokens, **kwargs):
        return "".join({3: "A", 4: "B", 5: "S", 6: "TOP", 9: ""}[t] for t in tokens)


class Runtime:
    def __init__(self):
        self.requests = {}
        self.completions = {}
        self.released = []
        self.abandoned = []
        self.shutdown = threading.Event()
    def reset(self): pass
    def start(self): pass
    def stop(self): pass
    def request_shutdown(self): self.shutdown.set()
    def submit(self, rid, tokens, generation_config):
        self.requests[rid] = generation_config
    def get_completion(self, rid):
        return self.completions.get(rid)
    def find_row_for_rid(self, rid): return -1
    def read_tokens_range(self, row, start, end): return torch.tensor([3, 5, 6, 4][:end-start+1])
    def finish_reason(self, row): return "length"
    def release_request(self, rid): self.released.append(rid)
    def abandon_request(self, rid): self.abandoned.append(rid)


@pytest.fixture
def engine():
    runtime = Runtime()
    class Runner:
        tokenizer = Tokenizer()
        vocab_size = 10
        eos_ids = [9]
        config = SimpleNamespace(developer_role="system", max_seq_length=20, max_pending_requests=2)
        def __call__(self): runtime.shutdown.wait()
    runner = Runner(); runner.runtime = runtime
    engine = LLMEngine(runner)
    yield engine
    engine.close()


def test_complete_stop_and_request_isolation(engine):
    first = SamplingParams(temperature=.3, seed=17, stop=("STOP",))
    second = SamplingParams(temperature=1, seed=42)
    a = engine.generate(*engine.prepare(prompt="a", params=first), first)
    b = engine.generate(*engine.prepare(prompt="b", params=second), second)
    assert engine.runtime.requests[a.rid][1] == 17
    assert engine.runtime.requests[b.rid][1] == 42
    engine.runtime.completions[a.rid] = (0, 5)
    engine.runtime.completions[b.rid] = (1, 5)
    ea, eb = list(a), list(b)
    assert "".join(e.text for e in ea) == "A"
    assert ea[-1].finish_reason == "stop"
    assert ea[-1].completion_tokens == 3
    assert "".join(e.text for e in eb) == "ASTOPB"
    assert eb[-1].finish_reason == "length"
    assert set(engine.runtime.released) == {a.rid, b.rid}


def test_timeout_and_cancel(engine):
    params = SamplingParams()
    session = engine.generate(*engine.prepare(prompt="a"), params, timeout=.01)
    with pytest.raises(TimeoutError): list(session)
    assert session.rid in engine.runtime.abandoned
    session = engine.generate(*engine.prepare(prompt="a"), params)
    session.close()
    assert session.rid in engine.runtime.abandoned
    with pytest.raises(RuntimeError, match="cancelled"): list(session)


def test_queue_limit(engine):
    params = SamplingParams()
    a = engine.generate(*engine.prepare(prompt="a"), params)
    b = engine.generate(*engine.prepare(prompt="b"), params)
    with pytest.raises(OverflowError): engine.generate(*engine.prepare(prompt="c"), params)
    a.close(); b.close()
