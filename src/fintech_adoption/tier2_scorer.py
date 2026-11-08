"""
tier2_scorer.py — Tier-2 phrase-level scoring.

For each candidate segment p_j from Tier 1:
  - Extract filtered n-gram candidates g_jl.
  - Compute seed proximity r_jl.
  - Compute intrinsic weight w_jl from TF-IDF, POS pattern, and proximity
    to adoption verbs.
  - Keep top key phrases as evidence for LLM adjudication and audit.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Tier2Config
from .embedder import FinTechEmbedder
from .preprocessor import Segment

logger = logging.getLogger(__name__)


VALUABLE_POS_PATTERNS = {
    ("NOUN",): 0.5,
    ("PROPN",): 0.6,
    ("ADJ", "NOUN"): 0.7,
    ("ADJ", "PROPN"): 0.7,
    ("NOUN", "NOUN"): 0.8,
    ("PROPN", "PROPN"): 0.8,
    ("VERB", "NOUN"): 0.9,
    ("VERB", "PROPN"): 0.9,
    ("ADJ", "NOUN", "NOUN"): 0.85,
    ("NOUN", "NOUN", "NOUN"): 0.85,
    ("PROPN", "NOUN", "NOUN"): 0.9,
    ("ADJ", "ADJ", "NOUN"): 0.75,
}

STOPWORDS_VI_EN = {
    "và", "của", "cho", "các", "những", "một", "này", "đó", "trong", "với",
    "đã", "đang", "sẽ", "là", "có", "the", "and", "of", "for", "to", "in",
    "on", "by", "a", "an", "has", "have", "is", "are",
}


def is_valid_phrase(phrase: str) -> bool:
    tokens = phrase.strip().split()
    if not tokens:
        return False
    if all(t.lower() in STOPWORDS_VI_EN for t in tokens):
        return False
    if sum(ch.isdigit() for ch in phrase) > max(2, len(phrase) // 2):
        return False
    if len(phrase) <= 2:
        return False
    return True


def pos_score_for_pattern(pos_sequence: Tuple[str, ...]) -> float:
    return VALUABLE_POS_PATTERNS.get(pos_sequence, 0.3)


def _build_adoption_verb_regex(verbs: List[str]) -> re.Pattern:
    escaped = [re.escape(v) for v in sorted(verbs, key=len, reverse=True)]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def dependency_score(
    phrase: str,
    segment_text: str,
    adoption_verb_pattern: re.Pattern,
    window_chars: int = 120,
) -> float:
    phrase_pos = segment_text.lower().find(phrase.lower())
    if phrase_pos < 0:
        return 0.1

    for match in adoption_verb_pattern.finditer(segment_text):
        distance = abs(match.start() - phrase_pos)
        if distance <= window_chars:
            return 1.0

    return 0.5


def extract_ngrams_basic(
    text: str,
    ngram_min: int = 1,
    ngram_max: int = 4,
) -> List[str]:
    """Fast regex-based n-gram extraction with basic noise filters."""
    tokens = re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", text, re.UNICODE)
    ngrams = []
    for n in range(ngram_min, ngram_max + 1):
        for i in range(len(tokens) - n + 1):
            gram = " ".join(tokens[i : i + n])
            if is_valid_phrase(gram):
                ngrams.append(gram)
    return list(set(ngrams))


def extract_ngrams_spacy(
    text: str,
    nlp,
    ngram_min: int = 1,
    ngram_max: int = 4,
    valid_pos_patterns: Optional[List[str]] = None,
) -> List[Tuple[str, Tuple[str, ...]]]:
    doc = nlp(text[:50_000])
    tokens = list(doc)
    if valid_pos_patterns:
        valid_tuples = set(tuple(p.split()) for p in valid_pos_patterns)
    else:
        valid_tuples = set(VALUABLE_POS_PATTERNS.keys())

    results = []
    for n in range(ngram_min, ngram_max + 1):
        for i in range(len(tokens) - n + 1):
            span = tokens[i : i + n]
            pos_seq = tuple(t.pos_ for t in span)
            phrase_text = " ".join(t.text for t in span)
            if (pos_seq in valid_tuples or not valid_pos_patterns) and is_valid_phrase(phrase_text):
                results.append((phrase_text, pos_seq))
    return results


class CorpusTFIDF:
    """Corpus-level TF-IDF scorer for candidate phrases."""

    def __init__(self, ngram_range: Tuple[int, int] = (1, 4)):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            analyzer="word",
            token_pattern=r"[\w\u00C0-\u024F\u1E00-\u1EFF]+",
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
        )
        self._fitted = False
        self._feature_names: Optional[np.ndarray] = None
        self._tfidf_matrix = None

    def fit(self, corpus: List[str]) -> None:
        self._tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self._feature_names = self.vectorizer.get_feature_names_out()
        self._fitted = True
        logger.info(f"[TF-IDF] Fitted on {len(corpus)} docs | vocab={len(self._feature_names):,}")

    def score_phrase_in_doc(self, phrase: str, doc_index: int) -> float:
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        phrase_lower = phrase.lower()
        feat_idx = np.where(self._feature_names == phrase_lower)[0]
        if len(feat_idx) == 0:
            return 0.0
        score = self._tfidf_matrix[doc_index, feat_idx[0]]
        return float(score)

    def score_phrase_raw(self, phrase: str, doc_text: str) -> float:
        phrase_lower = phrase.lower()
        doc_lower = doc_text.lower()
        tf = doc_lower.count(phrase_lower)
        if tf == 0:
            return 0.0
        total_terms = len(doc_lower.split())
        return (tf / max(total_terms, 1)) * np.log1p(1.0)


class Tier2Scorer:
    """Scores phrase-level candidates within top-K Tier-1 segments."""

    def __init__(
        self,
        embedder: FinTechEmbedder,
        cfg: Tier2Config,
        tfidf: Optional[CorpusTFIDF] = None,
        use_spacy: bool = False,
        spacy_model: str = "en_core_web_sm",
    ):
        self.embedder = embedder
        self.cfg = cfg
        self.tfidf = tfidf
        self.use_spacy = use_spacy
        self._nlp = None
        self._adoption_pattern = _build_adoption_verb_regex(
            cfg.adoption_verbs_vi + cfg.adoption_verbs_en
        )

        if use_spacy:
            try:
                import spacy

                self._nlp = spacy.load(spacy_model, disable=["ner", "lemmatizer"])
                logger.info(f"[Tier2] spaCy model '{spacy_model}' loaded.")
            except Exception as e:
                logger.warning(f"[Tier2] spaCy unavailable ({e}). Using basic n-grams.")
                self.use_spacy = False

    def score_segment(
        self,
        seg: Segment,
        doc_index: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        text = seg.text
        if self.use_spacy and self._nlp:
            phrase_pos_pairs = extract_ngrams_spacy(
                text,
                self._nlp,
                self.cfg.ngram_min,
                self.cfg.ngram_max,
                valid_pos_patterns=self.cfg.valid_pos_patterns,
            )
            phrases = [p for p, _ in phrase_pos_pairs]
            pos_sequences = [pos for _, pos in phrase_pos_pairs]
        else:
            phrases = extract_ngrams_basic(text, self.cfg.ngram_min, self.cfg.ngram_max)
            pos_sequences = [("NOUN",)] * len(phrases)

        if not phrases:
            return []

        phrase_embs = self.embedder.encode_phrases(phrases)
        r_jl = self.embedder.phrase_seed_similarity(phrase_embs)

        scores = []
        for i, (phrase, pos_seq) in enumerate(zip(phrases, pos_sequences)):
            if self.tfidf and self.tfidf._fitted:
                lookup_index = seg.tfidf_doc_index if doc_index is None else doc_index
                if lookup_index >= 0:
                    tfidf_val = self.tfidf.score_phrase_in_doc(phrase, lookup_index)
                else:
                    tfidf_val = 0.0
                if tfidf_val == 0.0:
                    tfidf_val = self.tfidf.score_phrase_raw(phrase, text)
            else:
                tfidf_val = self._simple_tfidf(phrase, text)

            pos_val = pos_score_for_pattern(pos_seq)
            dep_val = dependency_score(phrase, text, self._adoption_pattern)
            w_jl = (
                self.cfg.lambda1 * tfidf_val
                + self.cfg.lambda2 * pos_val
                + self.cfg.lambda3 * dep_val
            )
            scores.append((phrase, float(r_jl[i]) * w_jl))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def _simple_tfidf(self, phrase: str, doc_text: str) -> float:
        count = doc_text.lower().count(phrase.lower())
        total = max(len(doc_text.split()), 1)
        tf = count / total
        length_bonus = min(len(phrase.split()) / 4, 1.0)
        return tf * (1 + length_bonus)

    def score_segments(self, segments: List[Segment], top_n: int = 3) -> None:
        logger.info(f"[Tier2] Scoring phrases in {len(segments)} segments ...")
        for seg in segments:
            phrase_scores = self.score_segment(seg)
            if phrase_scores:
                top_phrases = phrase_scores[:top_n]
                seg.key_phrases = [p for p, _ in top_phrases]
                seg.S2_j = top_phrases[0][1]
            else:
                seg.S2_j = 0.0
                seg.key_phrases = []

        logger.info(f"[Tier2] Done | Avg S2={np.mean([s.S2_j for s in segments]):.4f}")
