"""OpenAI-compatible request models and backend-independent sampling params."""

from __future__ import annotations

import json
import math
import secrets
import struct
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

MAX_BIASES = 256
MAX_STOP_SEQUENCES = 4
MIN_LOGIT_BIAS = -100.0
MAX_LOGIT_BIAS = 100.0
CONFIG_WORDS = 544
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
            raise ValueError("tool_call_id is only required for tool messages")
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls requires the assistant role")
        return self

    def template_message(self):
        value = self.model_dump(exclude_none=True)
        if isinstance(self.content, list):
            value["content"] = "".join(part.text for part in self.content)

        for call in value.get("tool_calls", []):
            arguments = call["function"]["arguments"]

            if isinstance(arguments, str):
                try:
                    call["function"]["arguments"] = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "tool call arguments must be valid JSON"
                    ) from exc

        return value

class StreamOptions(APIModel):
    include_usage: bool = False

class CompletionRequest(APIModel):
    model: str
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: StrictInt = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    repetition_penalty: float = Field(default=1, gt=0)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**63 - 1)
    max_tokens: StrictInt | None = Field(default=None, gt=0)
    #max_completion_tokens: StrictInt | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    logit_bias: dict[str, float] = Field(default_factory=dict, max_length=MAX_BIASES)
    n: Literal[1] = 1
    user: str | None = None

    @model_validator(mode="after")
    def check_options(self):
        try:
            repetition = struct.unpack("f", struct.pack("f", self.repetition_penalty))[0]
        except OverflowError as exc:
            raise ValueError("repetition_penalty must fit a positive finite float32") from exc

        if repetition == 0 or not math.isfinite(repetition):
            raise ValueError("repetition_penalty must fit a positive finite float32")

        stops = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if len(stops) > MAX_STOP_SEQUENCES or any(not s for s in stops):
            raise ValueError("stop must contain one to four nonempty strings")

        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")

        for token, bias in self.logit_bias.items():
            if (
                not token.isdecimal()
                or not math.isfinite(bias)
                or not MIN_LOGIT_BIAS <= bias <= MAX_LOGIT_BIAS
            ):
                raise ValueError(
                    f"logit_bias requires nonnegative token IDs and finite values "
                    f"in [{MIN_LOGIT_BIAS}, {MAX_LOGIT_BIAS}]"
                )

        return self

    def sampling_params(self):
        return SamplingParams(
            temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
            frequency_penalty=self.frequency_penalty, presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed if self.seed is not None else secrets.randbits(63),
            max_tokens=self.max_tokens,
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

def _float_to_bits(value: float) -> int:
    """Encode a Python float as its float64 bit pattern in a signed int64."""
    return struct.unpack("q", struct.pack("d", value))[0]
 
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

    def pack(
        self,
        prompt_len: int,
        max_seq_length: int,
        vocab_size: int,
        eos_ids: list[int],
    ) -> list[int]:

        # 1. Determine output token budget.
        remaining = max_seq_length - prompt_len
        budget = (
            self.max_new_tokens
            if self.max_new_tokens is not None
            else remaining
        )

        # 2. Runtime/model-dependent validation.
        if prompt_len < 1:
            raise ValueError("prompt must contain at least one token")

        if budget < 1 or budget > remaining:
            raise ValueError(
                "prompt plus requested output exceeds context capacity"
            )

        if len(eos_ids) > MAX_EOS:
            raise ValueError("too many EOS token IDs")

        if any(token_id < 0 or token_id >= vocab_size for token_id in eos_ids):
            raise ValueError("invalid model EOS token ID")

        if any(
            token_id < 0 or token_id >= vocab_size
            for token_id in self.logit_bias
        ):
            raise ValueError(
                "logit_bias token ID exceeds model vocabulary"
            )

        # 3. Allocate fixed-size GPU config.
        words = [0] * CONFIG_WORDS

        # Header.
        words[0] = budget
        words[1] = self.seed
        words[2] = vocab_size
        words[3] = _float_to_bits(self.temperature)
        words[4] = _float_to_bits(self.top_p)
        words[5] = min(self.top_k, vocab_size)
        words[6] = _float_to_bits(self.frequency_penalty)
        words[7] = _float_to_bits(self.presence_penalty)
        words[8] = _float_to_bits(self.repetition_penalty)
        words[9] = len(eos_ids)
        words[10] = len(self.logit_bias)

        # EOS token IDs: words [16, 32)
        words[16 : 16 + len(eos_ids)] = eos_ids

        # Logit biases: words [32, 544)
        for i, (token_id, bias) in enumerate(
            sorted(self.logit_bias.items())
        ):
            offset = 32 + 2 * i
            words[offset] = token_id
            words[offset + 1] = _float_to_bits(bias)

        return words