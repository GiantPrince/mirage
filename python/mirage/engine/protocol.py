"""Validated text OpenAI protocol and backend-independent generation settings."""
from __future__ import annotations

import math
import secrets
import struct
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

# Shared ABI with tasks/common/serving_sampling.cuh. Doubles are bit-cast to i64.
CONFIG_WORDS = 544
MAX_BIASES = 256
MAX_EOS = 16


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextPart(APIModel):
    type: Literal["text"]
    text: str


class FunctionCall(APIModel):
    name: str
    arguments: str


class ToolCall(APIModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(APIModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def check_role(self):
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("content is required except for assistant tool calls")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool_call_id is required only for tool messages")
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls requires the assistant role")
        return self

    def template_message(self):
        value = self.model_dump(exclude_none=True)
        if isinstance(self.content, list):
            value["content"] = "".join(part.text for part in self.content)
        return value


class StreamOptions(APIModel):
    include_usage: bool = False


class CompletionRequest(APIModel):
    model: str
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: StrictInt = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    repetition_penalty: float = Field(default=1, gt=0)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**63 - 1)
    max_tokens: StrictInt | None = Field(default=None, gt=0)
    max_completion_tokens: StrictInt | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] = Field(default_factory=dict, max_length=MAX_BIASES)
    n: Literal[1] = 1
    user: str | None = None

    @model_validator(mode="after")
    def check_options(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            if self.max_tokens != self.max_completion_tokens:
                raise ValueError("max_tokens and max_completion_tokens conflict")
        try:
            repetition = struct.unpack("f", struct.pack("f", self.repetition_penalty))[0]
        except OverflowError as exc:
            raise ValueError("repetition_penalty must fit a positive finite float32") from exc
        if repetition == 0 or not math.isfinite(repetition):
            raise ValueError("repetition_penalty must fit a positive finite float32")
        stops = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if len(stops) > 4 or any(not s for s in stops):
            raise ValueError("stop must contain one to four nonempty strings")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        for token, bias in self.logit_bias.items():
            if not token.isdecimal() or not math.isfinite(bias) or not -100 <= bias <= 100:
                raise ValueError("logit_bias requires nonnegative token IDs and finite values in [-100, 100]")
        return self

    def sampling_params(self):
        return SamplingParams(
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            frequency_penalty=self.frequency_penalty, presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed if self.seed is not None else secrets.randbits(63),
            max_new_tokens=self.max_completion_tokens or self.max_tokens,
            stop=tuple([self.stop] if isinstance(self.stop, str) else self.stop or []),
            logit_bias={int(k): v for k, v in self.logit_bias.items()},
        )


class ChatRequest(CompletionRequest):
    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def check_tool_history(self):
        pending = set()
        seen = set()
        for message in self.messages:
            if pending and message.role != "tool":
                raise ValueError("assistant tool calls must be followed by their tool results")
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("tool result has no matching assistant tool call")
                pending.remove(message.tool_call_id)
            for call in message.tool_calls or []:
                if call.id in seen:
                    raise ValueError("duplicate tool call ID")
                pending.add(call.id)
                seen.add(call.id)
        if pending:
            raise ValueError("missing tool results")
        return self


class TextRequest(CompletionRequest):
    prompt: str


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 0.0  # Preserve greedy defaults for Python callers.
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int = 0
    max_new_tokens: int | None = None
    stop: tuple[str, ...] = ()
    logit_bias: dict[int, float] = field(default_factory=dict)

    def pack(self, prompt_len, max_seq_length, vocab_size, eos_ids):
        remaining = max_seq_length - prompt_len
        budget = self.max_new_tokens if self.max_new_tokens is not None else remaining
        if prompt_len < 1 or budget < 1 or budget > remaining:
            raise ValueError("prompt plus requested output exceeds the context capacity")
        # Validate Python callers through the same contract as HTTP callers.
        CompletionRequest(model="internal", temperature=self.temperature, top_p=self.top_p,
                          top_k=self.top_k, frequency_penalty=self.frequency_penalty,
                          presence_penalty=self.presence_penalty,
                          repetition_penalty=self.repetition_penalty, seed=self.seed,
                          max_tokens=budget, stop=list(self.stop),
                          logit_bias={str(k): v for k, v in self.logit_bias.items()})
        if len(eos_ids) > MAX_EOS or any(not 0 <= t < vocab_size for t in eos_ids):
            raise ValueError("invalid model EOS token IDs")
        if any(not 0 <= t < vocab_size for t in self.logit_bias):
            raise ValueError("logit_bias token ID exceeds model vocabulary")
        def bits(value):
            return struct.unpack("q", struct.pack("d", value))[0]
        words = [0] * CONFIG_WORDS
        words[:11] = [budget, self.seed, vocab_size, bits(self.temperature), bits(self.top_p),
                      min(self.top_k, vocab_size), bits(self.frequency_penalty), bits(self.presence_penalty),
                      bits(self.repetition_penalty), len(eos_ids), len(self.logit_bias)]
        words[16:16 + len(eos_ids)] = eos_ids
        for i, (token, bias) in enumerate(sorted(self.logit_bias.items())):
            words[32 + 2*i:34 + 2*i] = [token, bits(bias)]
        return words
