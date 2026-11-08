"""
llm_adjudicator.py - LLM adjudication for filtered FinTech-adoption segments.

The LLM sees only pre-filtered candidate passages and returns a structured
label. Only confirmed, bank-internal deployment should contribute to FTA_it.
Future plans and general discussion are retained for audit but receive zero
adoption probability in the default research specification.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

from .config import LLMAdjudicationConfig
from .preprocessor import Segment

logger = logging.getLogger(__name__)

class LLMPreflightError(RuntimeError):
    """Raised when the configured LLM cannot be reached before a pipeline run."""


SYSTEM_PROMPT = """
Your task is to classify whether a passage from a bank annual report provides
evidence that the bank has actually adopted a specific FinTech technology or
digital capability.

Classification rules:
- ADOPTED: The bank explicitly states it has deployed, implemented, launched,
  integrated, operated, or is actively using a specific FinTech technology.
- DISCUSSED_ONLY: The bank mentions a future plan, intention, industry trend,
  regulatory requirement, partnership under consideration, or aspiration without
  confirming actual deployment by the reporting bank.
- IRRELEVANT: The passage is not meaningful evidence of bank FinTech adoption.

Distinguish the bank adopting technology itself from merely acknowledging
FinTech in the market.

Respond only with valid JSON:
{
  "label": "ADOPTED" | "DISCUSSED_ONLY" | "IRRELEVANT",
  "adoption_probability": <float between 0.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0>,
  "fintech_type": "<specific technology or 'none'>",
  "adoption_stage": "irrelevant" | "discussed" | "planned" | "pilot" | "deployed" | "scaled",
  "bank_is_actor": <true|false>,
  "specific_capability": <true|false>,
  "future_or_plan_only": <true|false>,
  "evidence": "<one short sentence>",
  "reason": "<one short sentence>"
}
""".strip()

FUTURE_PLAN_PATTERNS = [
    "se ", "se du kien", "ke hoach", "du kien", "sap trien khai",
    "plan", "plans", "planned", "will", "intend", "intends",
    "intention", "considering", "expected to", "aims to",
]

ADOPTED_PATTERNS = [
    "trien khai", "ung dung", "ap dung", "tich hop", "ra mat", "van hanh",
    "su dung", "deployed", "launched", "implemented", "integrated",
    "adopted", "operates", "is using", "has used", "mobile banking",
    "ekyc", "digital onboarding",
]


def _strip_accents(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def build_user_prompt(seg: Segment) -> str:
    key_phrase_str = (
        ", ".join(f'"{p}"' for p in seg.key_phrases[:5])
        if seg.key_phrases
        else "none identified"
    )
    keyword = seg.keyword_features or {}
    diagnostics = {
        "matched_concept_ids": keyword.get("matched_concept_ids", []),
        "adoption_verb_count": keyword.get("adoption_verb_count", 0),
        "future_plan_marker_count": keyword.get("future_plan_marker_count", 0),
        "has_bank_self_action": keyword.get("has_bank_self_action", False),
        "has_specific_product_or_process": keyword.get("has_specific_product_or_process", False),
    }
    return f"""
Passage:
---
{seg.text}
---

Detected FinTech phrases: {key_phrase_str}
Keyword/adoption diagnostics: {json.dumps(diagnostics, ensure_ascii=False)}
""".strip()


def parse_llm_response_fields(
    raw_text: str,
    cfg: LLMAdjudicationConfig,
) -> Dict[str, object]:
    """Parse a JSON response into structured adjudication fields."""
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).strip()
    cleaned = cleaned.removesuffix("```").strip()

    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object.")
        label = str(data.get("label", cfg.label_irrelevant)).upper()
        if label not in {cfg.label_adopted, cfg.label_discussed, cfg.label_irrelevant}:
            label = cfg.label_irrelevant
        confidence = float(data.get("confidence", 0.0))
        if not math.isfinite(confidence):
            raise ValueError("Confidence must be finite.")
        confidence = max(0.0, min(1.0, confidence))
        adoption_probability = float(
            data.get("adoption_probability", confidence if label == cfg.label_adopted else 0.0)
        )
        if not math.isfinite(adoption_probability):
            raise ValueError("Adoption probability must be finite.")
        adoption_probability = max(0.0, min(1.0, adoption_probability))
        fintech_type = str(data.get("fintech_type", "none") or "none")
        adoption_stage = str(data.get("adoption_stage", "") or "").lower()
        flags = {
            "bank_is_actor": data.get("bank_is_actor", False),
            "specific_capability": data.get("specific_capability", False),
            "future_or_plan_only": data.get("future_or_plan_only", label == cfg.label_discussed),
        }
        if any(type(value) is not bool for value in flags.values()):
            raise ValueError("Adoption flags must be JSON booleans.")
        bank_is_actor = flags["bank_is_actor"]
        specific_capability = flags["specific_capability"]
        future_or_plan_only = flags["future_or_plan_only"]
        evidence = str(data.get("evidence", "") or "")
        reason = str(data.get("reason", "") or "")
        return {
            "label": label,
            "confidence": confidence,
            "adoption_probability": adoption_probability,
            "fintech_type": fintech_type,
            "adoption_stage": adoption_stage,
            "bank_is_actor": bank_is_actor,
            "specific_capability": specific_capability,
            "future_or_plan_only": future_or_plan_only,
            "evidence": evidence,
            "reason": reason,
        }
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning("[LLM] Parse error: %s | Raw: %s", exc, raw_text[:200])
        return {
            "label": cfg.label_irrelevant,
            "confidence": 0.1,
            "adoption_probability": 0.0,
            "fintech_type": "unknown",
            "adoption_stage": "irrelevant",
            "bank_is_actor": False,
            "specific_capability": False,
            "future_or_plan_only": False,
            "evidence": "Parse fallback: irrelevant.",
            "reason": "The response could not be parsed as JSON; failed closed.",
        }


def parse_llm_response(
    raw_text: str,
    cfg: LLMAdjudicationConfig,
) -> Tuple[str, float, str, str]:
    """Backward-compatible parser returning label, confidence, type, evidence."""
    fields = parse_llm_response_fields(raw_text, cfg)
    return (
        str(fields["label"]),
        float(fields["confidence"]),
        str(fields["fintech_type"]),
        str(fields["evidence"]),
    )


def call_anthropic(
    user_prompt: str,
    cfg: LLMAdjudicationConfig,
    api_key: Optional[str] = None,
) -> str:
    """Single Anthropic API call. Returns raw response text."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("pip install anthropic") from exc

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY not set.")

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return message.content[0].text


def call_openai(
    user_prompt: str,
    cfg: LLMAdjudicationConfig,
    api_key: Optional[str] = None,
    client_max_retries: Optional[int] = None,
) -> str:
    """Single OpenAI Responses API call. Returns raw response text."""
    try:
        import openai
    except ImportError as exc:
        raise ImportError("pip install openai") from exc

    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set.")

    client_kwargs = {"api_key": key}
    if client_max_retries is not None:
        client_kwargs["max_retries"] = client_max_retries
    client = openai.OpenAI(**client_kwargs)
    request = {
        "model": cfg.model,
        "instructions": SYSTEM_PROMPT,
        "input": f"Return JSON only.\n\n{user_prompt}",
        "max_output_tokens": cfg.max_tokens,
        "text": {"format": {"type": "json_object"}},
    }
    model_name = cfg.model.lower()
    if not model_name.startswith(("gpt-5", "codex", "o1", "o3", "o4")):
        request["temperature"] = cfg.temperature
    response = client.responses.create(**request)
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output_chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                output_chunks.append(text)
    return "\n".join(output_chunks)

def run_llm_preflight(
    cfg: LLMAdjudicationConfig,
    api_key: Optional[str] = None,
    mock: bool = False,
) -> Dict[str, str]:
    """
    Send one tiny structured request before expensive PDF/embedding work.

    This catches missing keys, quota/billing errors, unsupported model
    parameters, and JSON-mode request issues before the pipeline processes a
    long annual report.
    """
    if mock:
        return {"status": "skipped", "reason": "mock_llm"}

    prompt = (
        'Return JSON only: {"status":"ok"}. '
        "This is a preflight connectivity and quota check."
    )
    try:
        if cfg.provider == "anthropic":
            raw = call_anthropic(prompt, cfg, api_key)
        else:
            raw = call_openai(prompt, cfg, api_key, client_max_retries=0)
        parsed = json.loads(raw)
        return {"status": str(parsed.get("status", "ok"))}
    except Exception as exc:
        raise LLMPreflightError(
            "LLM preflight failed before PDF processing. "
            "Check API key, quota/billing, model name, and provider config. "
            f"Original error: {exc}"
        ) from exc


def call_mock(user_prompt: str, cfg: LLMAdjudicationConfig) -> str:
    """Deterministic local classifier for tests and dry runs."""
    # Diagnostic field names contain "future_plan" even for deployed passages.
    # Classify only the passage, keeping those field names out of the heuristic.
    passage = user_prompt.split("Passage:\n---\n", 1)[-1].split("\n---", 1)[0]
    text_lower = _strip_accents(passage.lower())

    if any(pattern in text_lower for pattern in FUTURE_PLAN_PATTERNS):
        return json.dumps({
            "label": cfg.label_discussed,
            "adoption_probability": 0.0,
            "confidence": 0.70,
            "fintech_type": "future_plan",
            "adoption_stage": "planned",
            "bank_is_actor": False,
            "specific_capability": False,
            "future_or_plan_only": True,
            "evidence": "Future-tense language indicates intention rather than confirmed adoption.",
            "reason": "The passage is framed as a plan rather than active deployment.",
        })

    if any(pattern in text_lower for pattern in ADOPTED_PATTERNS):
        return json.dumps({
            "label": cfg.label_adopted,
            "adoption_probability": 0.85,
            "confidence": 0.85,
            "fintech_type": "digital_banking",
            "adoption_stage": "deployed",
            "bank_is_actor": True,
            "specific_capability": True,
            "future_or_plan_only": False,
            "evidence": "The passage states a deployed or operating digital capability.",
            "reason": "The passage contains deployment or active-use language.",
        })

    return json.dumps({
        "label": cfg.label_irrelevant,
        "adoption_probability": 0.0,
        "confidence": 0.80,
        "fintech_type": "none",
        "adoption_stage": "irrelevant",
        "bank_is_actor": False,
        "specific_capability": False,
        "future_or_plan_only": False,
        "evidence": "No clear bank-level FinTech adoption signal was detected.",
        "reason": "The passage lacks a specific bank-level adoption signal.",
    })


class LLMAdjudicator:
    """Classify filtered candidate segments and write audit fields in-place."""

    def __init__(
        self,
        cfg: LLMAdjudicationConfig,
        api_key: Optional[str] = None,
        mock: bool = False,
        rate_limit_delay: float = 0.5,
        max_retries: int = 3,
    ):
        self.cfg = cfg
        self.api_key = api_key
        self.mock = mock
        self.rate_limit_delay = rate_limit_delay
        self.max_retries = max_retries

    def _call_api(self, user_prompt: str) -> str:
        if self.mock:
            return call_mock(user_prompt, self.cfg)

        for attempt in range(self.max_retries):
            try:
                if self.cfg.provider == "anthropic":
                    return call_anthropic(user_prompt, self.cfg, self.api_key)
                return call_openai(user_prompt, self.cfg, self.api_key)
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "[LLM] API error attempt %s/%s: %s. Retrying in %ss.",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        logger.error("[LLM] Max retries exceeded. Returning IRRELEVANT.")
        return json.dumps({
            "label": self.cfg.label_irrelevant,
            "confidence": 0.0,
            "fintech_type": "none",
            "evidence": "API failure.",
        })

    def _write_result(
        self,
        seg: Segment,
        label: str,
        confidence: float,
        fintech_type: str,
        evidence: str,
        raw_response: str,
        adoption_probability: float = 0.0,
        adoption_stage: str = "",
        bank_is_actor: bool = False,
        specific_capability: bool = False,
        future_or_plan_only: bool = False,
        reason: str = "",
    ) -> None:
        if label == self.cfg.label_adopted:
            guard_reasons = []
            if future_or_plan_only:
                guard_reasons.append("future_or_plan_only")
            if not bank_is_actor:
                guard_reasons.append("bank_is_actor_false")
            if not specific_capability:
                guard_reasons.append("specific_capability_false")
            if guard_reasons:
                label = self.cfg.label_discussed if future_or_plan_only else self.cfg.label_irrelevant
                adoption_probability = 0.0
                confidence = min(confidence, 0.4)
                reason = (
                    f"Guard downgraded ADOPTED because {', '.join(guard_reasons)}. "
                    f"{reason}"
                ).strip()

        seg.llm_label = label
        seg.llm_confidence_raw = confidence
        seg.llm_fintech_type = fintech_type
        seg.llm_evidence = evidence
        seg.llm_raw_response = raw_response
        seg.llm_adoption_probability = adoption_probability
        seg.llm_adoption_stage = adoption_stage
        seg.llm_bank_is_actor = bank_is_actor
        seg.llm_specific_capability = specific_capability
        seg.llm_future_or_plan_only = future_or_plan_only
        seg.llm_reason = reason
        label_probability = self.cfg.adoption_probability_by_label.get(label, 0.0)
        seg.P_llm_j = float(label_probability * adoption_probability)

    def adjudicate_segment(self, seg: Segment) -> None:
        """Classify one segment and write the result in-place."""
        user_prompt = build_user_prompt(seg)
        raw = self._call_api(user_prompt)
        fields = parse_llm_response_fields(raw, self.cfg)
        label = str(fields["label"])
        confidence = float(fields["confidence"])
        fintech_type = str(fields["fintech_type"])
        self._write_result(
            seg,
            label,
            confidence,
            fintech_type,
            str(fields["evidence"]),
            raw,
            adoption_probability=float(fields["adoption_probability"]),
            adoption_stage=str(fields["adoption_stage"]),
            bank_is_actor=bool(fields["bank_is_actor"]),
            specific_capability=bool(fields["specific_capability"]),
            future_or_plan_only=bool(fields["future_or_plan_only"]),
            reason=str(fields["reason"]),
        )

        logger.debug(
            "[LLM] Seg %s | %s | conf=%.2f | P=%.2f | %s",
            seg.seg_id,
            label,
            confidence,
            seg.P_llm_j,
            fintech_type,
        )
        time.sleep(self.rate_limit_delay)

    def adjudicate_segments(
        self,
        segments: List[Segment],
        skip_irrelevant_threshold: float = 0.15,
    ) -> None:
        """Classify all segments in-place, skipping low Tier-1 candidates."""
        n_skip = sum(1 for seg in segments if seg.S1_j < skip_irrelevant_threshold)
        logger.info(
            "[LLM] Adjudicating %s segments (%s skipped as low-score) ...",
            len(segments),
            n_skip,
        )

        for seg in segments:
            if seg.S1_j < skip_irrelevant_threshold:
                self._write_result(
                    seg,
                    self.cfg.label_irrelevant,
                    0.0,
                    "none",
                    "Skipped because Tier-1 relevance was below the LLM threshold.",
                    "",
                    adoption_stage="irrelevant",
                    reason="Skipped because Tier-1 relevance was below the LLM threshold.",
                )
                continue
            self.adjudicate_segment(seg)

        label_counts: Dict[str, int] = {}
        for seg in segments:
            label_counts[seg.llm_label] = label_counts.get(seg.llm_label, 0) + 1
        logger.info("[LLM] Label distribution: %s", label_counts)
