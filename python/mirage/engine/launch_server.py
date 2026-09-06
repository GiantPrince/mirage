"""OpenAI-compatible text generation server backed by Mirage's persistent kernel."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from .protocol import ChatRequest, TextRequest

logger = logging.getLogger(__name__)


def error_response(message, status=400, param=None, code=None):
    return JSONResponse(status_code=status, content={"error": {
        "message": message, "type": "invalid_request_error" if status < 500 else "server_error",
        "param": param, "code": code}})


@asynccontextmanager
async def lifespan(app):
    from .model_runner import ModelRunner
    from .llm_engine import LLMEngine
    app.state.engine = LLMEngine(ModelRunner(app.state.runner_config))
    try:
        yield
    finally:
        await asyncio.to_thread(app.state.engine.close)


def create_app(engine=None, *, model=None, request_timeout=120.0):
    app = FastAPI(title="Mirage OpenAI API", lifespan=lifespan if engine is None else None)
    app.state.engine = engine
    app.state.served_model = model
    app.state.request_timeout = request_timeout

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": app.state.served_model, "object": "model",
                                           "created": 0, "owned_by": "mirage"}]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await complete(request, chat=True)

    @app.post("/v1/completions")
    async def text(request: Request):
        return await complete(request, chat=False)

    return app


def _next(iterator):
    # StopIteration must not propagate through an asyncio Future.
    return next(iterator, None)


def _response_base(request_id, created, model, chat, stream=False):
    return dict(id=request_id, created=created, model=model,
                object="chat.completion.chunk" if chat and stream else
                       "chat.completion" if chat else "text_completion")


async def complete(request, chat):
    try:
        body = await request.json()
        req = (ChatRequest if chat else TextRequest).model_validate(body)
    except ValidationError as exc:
        first = exc.errors(include_input=False)[0]
        return error_response(first["msg"], param=".".join(map(str, first["loc"])))
    except (ValueError, UnicodeDecodeError):
        return error_response("Invalid or empty JSON body")
    model = request.app.state.served_model
    if req.model != model:
        return error_response(f"Model '{req.model}' is not served", 404, "model", "model_not_found")
    engine = request.app.state.engine
    params = req.sampling_params()
    try:
        kwargs = {"messages": [m.template_message() for m in req.messages]} if chat else {"prompt": req.prompt}
        ids, packed = await asyncio.to_thread(engine.prepare, params=params, **kwargs)
        submission = asyncio.create_task(asyncio.to_thread(
            engine.generate, ids, packed, params, request.app.state.request_timeout))
        try:
            session = await asyncio.shield(submission)
        except asyncio.CancelledError:
            # Publishing may already be running in a worker thread. Recover its
            # handle before propagating cancellation so no request is orphaned.
            try:
                abandoned = await submission
                abandoned.close()
            finally:
                raise
    except ValueError as exc:
        return error_response(str(exc))
    except OverflowError as exc:
        return error_response(str(exc), 429, code="server_overloaded")
    except Exception:
        logger.exception("Failed to submit generation")
        return error_response("Unable to start generation", 503)

    request_id = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
    created = int(time.time())
    iterator = iter(session)
    base = _response_base(request_id, created, model, chat, req.stream)

    async def events():
        try:
            while True:
                event = await asyncio.to_thread(_next, iterator)
                if event is None:
                    break
                yield event
        finally:
            # Session cancellation is thread-safe, even while next() is blocked.
            session.close()

    if req.stream:
        async def sse():
            def encode(value):
                return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"
            try:
                if chat:
                    yield encode({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                                       "finish_reason": None}]})
                async for event in events():
                    if event.text:
                        choice = {"index": 0, "finish_reason": None}
                        choice.update({"delta": {"content": event.text}} if chat else {"text": event.text})
                        yield encode({**base, "choices": [choice]})
                    if event.finish_reason is not None:
                        choice = {"index": 0, "finish_reason": event.finish_reason}
                        choice.update({"delta": {}} if chat else {"text": ""})
                        yield encode({**base, "choices": [choice]})
                        if req.stream_options and req.stream_options.include_usage:
                            yield encode({**base, "choices": [], "usage": event.usage})
                yield "data: [DONE]\n\n"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Streaming generation failed")
                message = "Generation timed out" if isinstance(exc, TimeoutError) else "Generation failed"
                yield encode({"error": {"message": message, "type": "server_error", "param": None, "code": None}})
                yield "data: [DONE]\n\n"
            finally:
                session.close()
        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def collect():
        result = ""
        final = None
        async for event in events():
            result += event.text
            final = event
        if final is None or final.finish_reason is None:
            raise RuntimeError("generation ended without a terminal event")
        choice = {"index": 0, "finish_reason": final.finish_reason}
        choice.update({"message": {"role": "assistant", "content": result}} if chat else {"text": result})
        return {**base, "choices": [choice], "usage": final.usage}

    task = asyncio.create_task(collect())
    try:
        while not task.done():
            await asyncio.wait({task}, timeout=0.05)
            if await request.is_disconnected():
                task.cancel()
                return error_response("Client disconnected", 499)
        return await task
    except TimeoutError:
        return error_response("Generation timed out", 504)
    except Exception:
        logger.exception("Generation failed")
        return error_response("Generation failed", 500)
    finally:
        session.close()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


app = create_app()


def main():
    from .model_runner import RunnerConfig
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--model-path")
    parser.add_argument("--served-model-name")
    for name, default in [("max-num-batched-requests", 4), ("max-num-batched-tokens", 8),
                          ("max-seq-length", 512), ("max-num-pages", 16), ("page-size", 4096),
                          ("pinned-ring-capacity", 8), ("max-pending-requests", 128)]:
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--developer-role", choices=["system", "native", "reject"], default="system",
                        help="Explicit model adapter; system maps developer messages to system messages")
    parser.add_argument("--output-dir")
    parser.add_argument("--request-timeout", type=float, default=120)
    args = parser.parse_args()
    config_keys = RunnerConfig.__dataclass_fields__
    app.state.runner_config = RunnerConfig(**{k: v for k, v in vars(args).items() if k in config_keys})
    app.state.served_model = args.served_model_name or args.model
    app.state.request_timeout = args.request_timeout
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
