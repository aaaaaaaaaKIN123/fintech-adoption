import json
import sys
from types import SimpleNamespace

import pytest

from fintech_adoption.config import LLMAdjudicationConfig
from fintech_adoption.llm_adjudicator import (
    LLMAdjudicator,
    LLMPreflightError,
    call_openai,
    parse_llm_response_fields,
    parse_llm_response,
    run_llm_preflight,
)
from fintech_adoption.preprocessor import Segment


@pytest.mark.parametrize("raw", ["[]", "null", '"ADOPTED"'])
def test_non_object_json_fails_closed(raw):
    fields = parse_llm_response_fields(raw, LLMAdjudicationConfig())
    assert fields["label"] == "IRRELEVANT"
    assert fields["adoption_probability"] == 0.0


@pytest.mark.parametrize("flag", ["bank_is_actor", "specific_capability", "future_or_plan_only"])
@pytest.mark.parametrize("value", ["false", "true", 1, None])
def test_non_boolean_flags_fail_closed(flag, value):
    response = {
        "label": "ADOPTED", "confidence": 0.9, "adoption_probability": 0.9,
        "bank_is_actor": True, "specific_capability": True, "future_or_plan_only": False,
    }
    response[flag] = value
    fields = parse_llm_response_fields(json.dumps(response), LLMAdjudicationConfig())
    assert fields["label"] == "IRRELEVANT"
    assert fields["adoption_probability"] == 0.0


@pytest.mark.parametrize("field", ["confidence", "adoption_probability"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_probabilities_fail_closed(field, value):
    response = {"label": "ADOPTED", "confidence": 0.9, "adoption_probability": 0.9}
    response[field] = value
    fields = parse_llm_response_fields(json.dumps(response), LLMAdjudicationConfig())
    assert fields["label"] == "IRRELEVANT"
    assert fields["adoption_probability"] == 0.0


def test_parse_response_keeps_raw_confidence_and_evidence():
    raw = json.dumps(
        {
            "label": "ADOPTED",
            "confidence": 0.72,
            "fintech_type": "eKYC",
            "evidence": "The bank states it launched eKYC.",
        }
    )

    label, confidence, fintech_type, evidence = parse_llm_response(raw, LLMAdjudicationConfig())

    assert label == "ADOPTED"
    assert confidence == 0.72
    assert fintech_type == "eKYC"
    assert "launched eKYC" in evidence


def test_discussed_only_has_zero_actual_adoption_probability():
    seg = Segment(
        seg_id=1,
        text="The bank plans to deploy open banking next year.",
        page_num=1,
        char_offset=0,
    )
    adjudicator = LLMAdjudicator(cfg=LLMAdjudicationConfig(), mock=True, rate_limit_delay=0.0)

    adjudicator.adjudicate_segment(seg)

    assert seg.llm_label == "DISCUSSED_ONLY"
    assert seg.P_llm_j == 0.0
    assert seg.llm_confidence_raw > 0.0
    assert seg.llm_evidence


def test_mock_classifies_deployed_passage_without_matching_diagnostic_field_names():
    seg = Segment(seg_id=1, text="The bank has deployed eKYC for online account opening.",
                  page_num=1, char_offset=0)
    adjudicator = LLMAdjudicator(cfg=LLMAdjudicationConfig(), mock=True, rate_limit_delay=0.0)
    adjudicator.adjudicate_segment(seg)
    assert seg.llm_label == "ADOPTED"
    assert seg.P_llm_j > 0

def test_malformed_llm_response_fails_closed_even_if_it_mentions_adopted():
    fields = parse_llm_response_fields(
        "ADOPTED: the bank launched mobile banking",
        LLMAdjudicationConfig(),
    )

    assert fields["label"] == "IRRELEVANT"
    assert fields["adoption_probability"] == 0.0
    assert "parsed" in fields["reason"].lower()

def test_adopted_response_without_bank_actor_or_capability_is_zeroed():
    class ContradictoryAdjudicator(LLMAdjudicator):
        def _call_api(self, user_prompt: str) -> str:
            return json.dumps(
                {
                    "label": "ADOPTED",
                    "adoption_probability": 0.9,
                    "confidence": 0.9,
                    "fintech_type": "market trend",
                    "adoption_stage": "deployed",
                    "bank_is_actor": False,
                    "specific_capability": False,
                    "future_or_plan_only": False,
                    "evidence": "The passage discusses market adoption.",
                    "reason": "The bank is not the actor.",
                }
            )

    seg = Segment(
        seg_id=2,
        text="The market adopted mobile banking widely.",
        page_num=1,
        char_offset=0,
    )
    adjudicator = ContradictoryAdjudicator(
        cfg=LLMAdjudicationConfig(),
        mock=False,
        rate_limit_delay=0.0,
    )

    adjudicator.adjudicate_segment(seg)

    assert seg.llm_label == "IRRELEVANT"
    assert seg.P_llm_j == 0.0
    assert "guard" in seg.llm_reason.lower()

def test_default_llm_adjudication_model_is_openai_codex():
    cfg = LLMAdjudicationConfig()

    assert cfg.provider == "openai"
    assert cfg.model == "gpt-5.5"

def test_openai_adjudication_uses_configured_codex_model(monkeypatch):
    captured = {}
    raw = json.dumps(
        {
            "label": "ADOPTED",
            "confidence": 0.9,
            "fintech_type": "mobile banking",
            "evidence": "The passage states the bank launched mobile banking.",
        }
    )

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text=raw)

    class FakeOpenAI:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            captured["client_kwargs"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    cfg = LLMAdjudicationConfig(model="gpt-5.5", provider="openai")
    response = call_openai("Passage: launched mobile banking.", cfg, api_key="test-key")

    assert json.loads(response)["label"] == "ADOPTED"
    assert captured["model"] == "gpt-5.5"
    assert captured["instructions"]
    assert "json" in captured["input"].lower()
    assert "Passage: launched mobile banking." in captured["input"]
    assert "temperature" not in captured

def test_llm_preflight_sends_one_small_openai_json_request(monkeypatch):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(output_text='{"status":"ok"}')

    class FakeOpenAI:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            captured["client_kwargs"] = kwargs
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    result = run_llm_preflight(
        LLMAdjudicationConfig(model="gpt-5.5", provider="openai"),
        api_key="test-key",
    )

    assert result["status"] == "ok"
    assert captured["model"] == "gpt-5.5"
    assert captured["client_kwargs"] == {"max_retries": 0}
    assert "json" in captured["input"].lower()
    assert "temperature" not in captured

def test_llm_preflight_raises_clear_error_on_quota_failure(monkeypatch):
    class FakeResponses:
        def create(self, **kwargs):
            raise RuntimeError("insufficient_quota")

    class FakeOpenAI:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key
            self.responses = FakeResponses()

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))

    try:
        run_llm_preflight(
            LLMAdjudicationConfig(model="gpt-5.5", provider="openai"),
            api_key="test-key",
        )
    except LLMPreflightError as exc:
        assert "LLM preflight failed" in str(exc)
        assert "insufficient_quota" in str(exc)
    else:
        raise AssertionError("Expected LLMPreflightError")
