import json
import pytest
from fastapi.testclient import TestClient
from mirage.engine.launch_server import create_app
from mirage.engine.output import GenerationEvent, OutputProcessor


class Session:
    closed = False
    def __iter__(self):
        yield GenerationEvent("hello ", prompt_tokens=7, completion_tokens=1)
        yield GenerationEvent("world", "length", 7, 2)
    def close(self):
        self.closed = True


class Engine:
    def prepare(self, **kwargs):
        self.prepared = kwargs
        return [1]*7, []
    def generate(self, *args):
        self.session = Session()
        return self.session


@pytest.fixture
def client():
    engine = Engine()
    with TestClient(create_app(engine, model="test")) as client:
        client.engine = engine
        yield client


def test_chat(client):
    messages = [{"role": "system", "content": "brief"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hey"}, {"role": "user", "content": "again"}]
    r = client.post("/v1/chat/completions", json=dict(model="test", messages=messages, temperature=.3))
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello world"
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"] == dict(prompt_tokens=7, completion_tokens=2, total_tokens=9)
    assert client.engine.prepared["messages"] == messages
    assert client.engine.prepared["params"].temperature == .3
    assert client.engine.session.closed


@pytest.mark.parametrize("chat", [False, True])
def test_stream(client, chat):
    payload = dict(model="test", stream=True, stream_options={"include_usage": True})
    payload.update(dict(messages=[dict(role="user", content="hi")]) if chat else dict(prompt="hi"))
    endpoint = "/v1/chat/completions" if chat else "/v1/completions"
    r = client.post(endpoint, json=payload)
    assert r.status_code == 200
    lines = [line[6:] for line in r.text.splitlines() if line.startswith("data: ")]
    assert lines.pop() == "[DONE]"
    chunks = [json.loads(line) for line in lines]
    assert len({c["id"] for c in chunks}) == 1
    assert all(c["model"] == "test" for c in chunks)
    assert chunks[-1]["usage"]["total_tokens"] == 9
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    if chat:
        assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    else:
        assert all("delta" not in c["choices"][0] for c in chunks[:-1])
        assert "prompt" in client.engine.prepared
    assert client.engine.session.closed


@pytest.mark.parametrize("payload,status", [(None, 400), ([], 400),
    ({"model": "missing", "prompt": "hi"}, 404),
    ({"model": "test", "prompt": "hi", "temperature": -1}, 400),
    ({"model": "test", "prompt": "hi", "logprobs": 3}, 400)])
def test_errors(client, payload, status):
    r = client.post("/v1/completions", json=payload)
    assert r.status_code == status
    assert "message" in r.json()["error"]


def test_models_and_unique_ids(client):
    assert client.get("/v1/models").json()["data"][0]["id"] == "test"
    ids = [client.post("/v1/completions", json=dict(model="test", prompt="hi")).json()["id"] for _ in range(2)]
    assert ids[0] != ids[1]


class Tokenizer:
    def decode(self, ids):
        return b"".join([b"a", b"ST", b"OP", b"extra", b"\xe4", b"\xbd\xa0"][i] for i in ids).decode("utf8", errors="replace")


def test_stop_across_tokens():
    proc = OutputProcessor(Tokenizer(), ("STOP",))
    assert proc.push(0) == "a"
    assert proc.push(1) == ""
    assert proc.push(2) == ""
    assert proc.stopped
    assert proc.push(3) == ""
    assert proc.token_ids == [0, 1, 2]
    assert proc.flush() == ""


def test_unicode_and_partial_stop_flush():
    proc = OutputProcessor(Tokenizer(), ("STOP",))
    assert proc.push(4) == ""
    assert proc.push(5) == "你"
    assert proc.push(1) == ""
    assert proc.flush() == "ST"
