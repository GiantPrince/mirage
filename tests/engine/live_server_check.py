"""Live persistent-kernel integration check. Requires built Mirage and a CUDA GPU."""
import concurrent.futures
import faulthandler
import json
import time

from fastapi.testclient import TestClient
from mirage.engine import LLMEngine, ModelRunner, RunnerConfig
from mirage.engine.launch_server import create_app


def main():
    faulthandler.dump_traceback_later(60, repeat=True)
    runner = ModelRunner(RunnerConfig(
        model="Qwen/Qwen3-0.6B", max_num_batched_requests=2,
        max_num_batched_tokens=8, max_seq_length=256, max_num_pages=8,
        page_size=64, output_dir="/tmp/mirage-openai-kernel"))
    engine = LLMEngine(runner)
    try:
        with TestClient(create_app(engine, model=runner.config.model, request_timeout=120)) as client:
            tokenizer = engine.tokenizer_manager._tokenizer
            forced = tokenizer.encode("Hello", add_special_tokens=False)[0]
            base = dict(model=runner.config.model, temperature=0, max_tokens=3,
                        logit_bias={str(forced): 100})
            response = client.post("/v1/chat/completions", json={**base, "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "My name is Ada."},
                {"role": "assistant", "content": "Hello Ada."},
                {"role": "developer", "content": "Remember the user's name."},
                {"role": "user", "content": "What is my name?"}]})
            assert response.status_code == 200, response.text
            data = response.json()
            assert data["usage"]["completion_tokens"] == 3, data
            assert data["choices"][0]["finish_reason"] == "length", data
            assert data["choices"][0]["message"]["content"] == tokenizer.decode([forced]*3), data
            print("history, developer role, bias and output budget: PASS", flush=True)

            def request(i, stream=False):
                payload = dict(model=runner.config.model, prompt="Hello", max_tokens=2 + i % 3,
                               temperature=.5 if i % 2 else 0, top_p=.8, seed=42 + i,
                               stream=stream, logit_bias={str(forced): 100})
                if stream:
                    payload["stream_options"] = dict(include_usage=True)
                result = client.post("/v1/completions", json=payload)
                assert result.status_code == 200, result.text
                if stream:
                    chunks = [json.loads(line[6:]) for line in result.text.splitlines()
                              if line.startswith("data: ") and line != "data: [DONE]"]
                    content = "".join(c["choices"][0].get("text", "") for c in chunks if c["choices"])
                    usage = chunks[-1]["usage"]
                else:
                    content = result.json()["choices"][0]["text"]
                    usage = result.json()["usage"]
                assert usage["completion_tokens"] == 2 + i % 3, result.text
                assert content == tokenizer.decode([forced] * (2 + i % 3)), result.text
                return content
            started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                result = list(pool.map(request, range(12)))
            assert request(1, stream=True) == result[1]
            print(f"12 concurrent requests, ring wrap, row reuse, mixed sampling, stream parity: PASS ({time.monotonic()-started:.2f}s)", flush=True)
            stopped = client.post("/v1/completions", json={**base, "prompt": "Hi", "stop": "Hello"})
            assert stopped.status_code == 200, stopped.text
            assert stopped.json()["choices"][0]["text"] == "", stopped.text
            assert stopped.json()["choices"][0]["finish_reason"] == "stop", stopped.text
            eos = engine.eos_ids[0]
            ended = client.post("/v1/completions", json={**base, "prompt": "Hi", "logit_bias": {str(eos): 100}})
            assert ended.json()["choices"][0]["finish_reason"] == "stop", ended.text
            assert ended.json()["usage"]["completion_tokens"] == 1, ended.text
            print("stop-string and EOS termination: PASS", flush=True)
            # Cross a KV page boundary and exhaust the exact context capacity.
            text = "Hello " * 70
            ids = tokenizer.encode(text, add_special_tokens=False)
            boundary = client.post("/v1/completions", json={**base, "prompt": text,
                                  "max_tokens": runner.config.max_seq_length - len(ids)})
            assert boundary.status_code == 200, boundary.text
            assert boundary.json()["usage"]["total_tokens"] == runner.config.max_seq_length, boundary.text
            print("KV-page and context-capacity boundaries: PASS", flush=True)
    finally:
        engine.close()
    print("LIVE SERVER CHECK PASSED", flush=True)


if __name__ == "__main__":
    main()
