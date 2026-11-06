import pytest

from retrieval.multilingual import (
    LLMTranslator, answer_language_instruction, detect_language,
    language_from_code, prepare_query,
)


@pytest.mark.parametrize("text, code", [
    ("What is the punishment for murder?", "en"),
    ("BNS section 103", "en"),
    ("हत्या की सजा क्या है?", "hi"),
    ("निजता का अधिकार क्या है?", "hi"),
    ("खुनाची शिक्षा काय आहे?", "mr"),
    ("கொலைக்கான தண்டனை என்ன?", "ta"),
    ("খুনের শাস্তি কী?", "bn"),
    ("", "en"),
    ("12345 ???", "en"),
])
def test_detect_language(text, code):
    assert detect_language(text).code == code


def test_mixed_text_mostly_english_stays_english():
    assert detect_language("What is the meaning of हत्या under IPC section 302?").code == "en"


def test_explicit_language_overrides_detection():
    assert language_from_code("MR").code == "mr"
    assert language_from_code("xx") is None
    assert language_from_code(None) is None


class FakeClient:
    def __init__(self, reply="What is the punishment for murder?", fail=False):
        self.calls, self.reply, self.fail = [], reply, fail
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("boom")

        class Msg:  # noqa: D401
            content = self.reply

        class Choice:
            message = Msg()

        class Resp:
            choices = [Choice()]

        return Resp()


def test_translator_translates_and_caches():
    client = FakeClient()
    t = LLMTranslator(client, "m")
    info = detect_language("हत्या की सजा क्या है?")
    assert t.to_english("हत्या की सजा क्या है?", info) == "What is the punishment for murder?"
    t.to_english("हत्या की सजा क्या है?", info)
    assert len(client.calls) == 1
    assert "Hindi" in client.calls[0]["messages"][1]["content"]


def test_translator_falls_back_to_original_on_error():
    t = LLMTranslator(FakeClient(fail=True), "m")
    q = "हत्या की सजा क्या है?"
    assert t.to_english(q, detect_language(q)) == q


def test_prepare_query_english_is_untouched():
    client = FakeClient()
    q, info = prepare_query("What is theft?", LLMTranslator(client, "m"))
    assert q == "What is theft?" and info.is_english and not client.calls


def test_prepare_query_without_translator_keeps_original():
    q, info = prepare_query("हत्या की सजा क्या है?", None)
    assert q == "हत्या की सजा क्या है?" and info.code == "hi"


def test_answer_language_instruction():
    assert answer_language_instruction(language_from_code("en")) == ""
    text = answer_language_instruction(language_from_code("mr"))
    assert "Marathi" in text and "[Source N]" in text
