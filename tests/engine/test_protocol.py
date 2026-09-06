import pytest
from pydantic import ValidationError
from mirage.engine.protocol import ChatRequest, TextRequest, SamplingParams
from mirage.engine.tokenizer_manager import TokenizerManager


def test_history_and_text_parts():
    req = ChatRequest(model="m", messages=[
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "My name is Ada"},
        {"role": "assistant", "content": "Hello Ada"},
        {"role": "user", "content": [{"type": "text", "text": "My name?"}]}])
    class Tokenizer:
        chat_template = "template"
        def apply_chat_template(self, messages, **kwargs):
            assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
            assert messages[0]["content"] == "Be brief"
            assert messages[-1]["content"] == "My name?"
            assert kwargs == dict(tokenize=True, add_generation_prompt=True)
            return [1, 2, 3]
    assert TokenizerManager(Tokenizer()).tokenize_messages(
        [m.template_message() for m in req.messages]) == [1, 2, 3]


@pytest.mark.parametrize("options", [dict(temperature=-1), dict(top_p=0), dict(top_k=1.5),
    dict(stop=""), dict(stop=["a"]*5), dict(n=2), dict(temperature=float("nan")),
    dict(max_tokens=1, max_completion_tokens=2), dict(logit_bias={"x": 1}),
    dict(stream_options={"include_usage": True}), dict(tools=[])])
def test_reject_invalid(options):
    with pytest.raises(ValidationError):
        TextRequest(model="m", prompt="hi", **options)


def test_context_and_bias_validation():
    with pytest.raises(ValueError):
        SamplingParams(max_new_tokens=3).pack(8, 10, 100, [2])
    with pytest.raises(ValueError):
        SamplingParams(logit_bias={100: 1}).pack(8, 10, 100, [2])
    words = SamplingParams(max_new_tokens=2, seed=17).pack(8, 10, 100, [2, 3])
    assert len(words) == 544
    assert words[:3] == [2, 17, 100]
    assert words[16:18] == [2, 3]


def test_tool_validation():
    with pytest.raises(ValidationError):
        ChatRequest(model="m", messages=[{"role": "tool", "content": "x", "tool_call_id": "a"}])
    ChatRequest(model="m", messages=[
        {"role": "assistant", "tool_calls": [{"id": "a", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "ok"}])
