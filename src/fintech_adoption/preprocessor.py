"""
preprocessor.py — PDF text extraction, paragraph segmentation, and
bilingual canonical text construction.

Responsibilities
────────────────
  1. Extract raw text from annual-report PDFs.
  2. Clean boilerplate: page numbers, headers, footers, obvious noise.
  3. Segment into paragraph-level units p_j with metadata.
  4. Detect Vietnamese/English/mixed segments.
  5. Remove bilingual duplicate paragraphs so FTA_it is not double-counted.
  6. Return a BankReport dataclass ready for the embedding pipeline.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .document_reader import DocumentReaderConfig, read_pdf_auto

logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """A paragraph-level text segment p_j."""

    seg_id: int
    text: str
    page_num: int
    char_offset: int
    word_count: int = 0

    language: str = ""
    canonical_seg_id: int = -1
    duplicate_seg_ids: List[int] = field(default_factory=list)
    languages_in_cluster: List[str] = field(default_factory=list)
    duplicate_similarity: float = 0.0
    tfidf_doc_index: int = -1

    nearest_seed_ids: List[str] = field(default_factory=list)
    nearest_seed_scores: List[float] = field(default_factory=list)

    R_j: float = 0.0
    C_j: float = 0.0
    S1_j: float = 0.0
    S2_j: float = 0.0

    llm_label: str = ""
    llm_confidence_raw: float = 0.0
    llm_fintech_type: str = ""
    llm_evidence: str = ""
    llm_raw_response: str = ""
    P_llm_j: float = 0.0
    llm_adoption_probability: float = 0.0
    llm_adoption_stage: str = ""
    llm_bank_is_actor: bool = False
    llm_specific_capability: bool = False
    llm_future_or_plan_only: bool = False
    llm_reason: str = ""

    S_j: float = 0.0
    key_phrases: List[str] = field(default_factory=list)
    keyword_features: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self.word_count = len(self.text.split())
        if not self.language:
            self.language = detect_language(self.text)
        if self.canonical_seg_id < 0:
            self.canonical_seg_id = self.seg_id
        if not self.languages_in_cluster:
            self.languages_in_cluster = [self.language]


@dataclass
class BankReport:
    """All canonical segments from one annual report of bank i at year t."""

    bank_id: str
    year: int
    pdf_path: str
    segments: List[Segment] = field(default_factory=list)
    clean_total_words: int = 0
    raw_total_words: int = 0
    canonical_clean_words: int = 0
    retained_segment_words: int = 0
    total_words: int = 0
    duplicate_segments: List[Dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.retained_segment_words = sum(s.word_count for s in self.segments)
        if self.canonical_clean_words <= 0:
            self.canonical_clean_words = self.retained_segment_words
        if self.clean_total_words <= 0:
            self.clean_total_words = self.canonical_clean_words
        if self.raw_total_words <= 0:
            self.raw_total_words = self.clean_total_words
        self.total_words = self.canonical_clean_words


_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*[\-–—]?\s*\d{1,4}\s*[\-–—]?\s*$", re.MULTILINE
)
_HEADER_FOOTER_PATTERN = re.compile(
    r"(BÁO CÁO THƯỜNG NIÊN|ANNUAL REPORT|BÁO CÁO TÀI CHÍNH|"
    r"FINANCIAL STATEMENTS|NỘI DUNG|TABLE OF CONTENTS|"
    r"Trang \d+|Page \d+)",
    re.IGNORECASE,
)
_EXCESSIVE_WHITESPACE = re.compile(r"\s{3,}")
_BULLET_NOISE = re.compile(r"^[\•\-\–\—\*\►\▪\◆\○\●]\s*", re.MULTILINE)

_VI_STOPWORDS = {
    "ngân", "hàng", "đã", "đang", "sẽ", "triển", "khai", "ứng", "dụng",
    "khách", "cá", "nhân", "trong", "năm", "cho", "với", "và", "của",
}
_EN_STOPWORDS = {
    "the", "bank", "has", "have", "will", "plans", "implemented", "launched",
    "deployed", "for", "with", "customers", "individual", "in", "and", "of",
}
_VI_DIACRITIC_CHARS = set(
    "ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ"
    "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ"
)

_TECH_PATTERNS = {
    "ekyc": r"\bekyc\b|định danh điện tử|định danh khách hàng",
    "mobile_banking": r"mobile banking|ngân hàng di động|ứng dụng ngân hàng|banking app",
    "open_banking": r"open banking|ngân hàng mở|api",
    "qr_payment": r"qr|vietqr|thanh toán qr",
    "ai": r"\bai\b|artificial intelligence|trí tuệ nhân tạo|machine learning|học máy",
    "cloud": r"cloud|đám mây",
    "blockchain": r"blockchain|chuỗi khối",
    "rpa": r"\brpa\b|robotic process automation|tự động hóa quy trình",
}
_ACTION_PATTERN = re.compile(
    r"triển khai|ứng dụng|áp dụng|tích hợp|ra mắt|vận hành|"
    r"implemented|deployed|launched|integrated|adopted|operates",
    re.IGNORECASE,
)


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def remove_boilerplate(text: str) -> str:
    text = _PAGE_NUMBER_PATTERN.sub("", text)
    text = _HEADER_FOOTER_PATTERN.sub("", text)
    text = _BULLET_NOISE.sub("", text)
    text = _EXCESSIVE_WHITESPACE.sub(" ", text)
    return text.strip()


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_boilerplate(text)
    text = re.sub(r"(?m)^\s*[a-zA-Z0-9]\s*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_language(text: str) -> str:
    """Heuristic language tag for Vietnamese bank disclosures."""
    lower = text.lower()
    tokens = set(re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", lower, re.UNICODE))
    vi_score = len(tokens & _VI_STOPWORDS)
    en_score = len(tokens & _EN_STOPWORDS)
    if any(ch.lower() in _VI_DIACRITIC_CHARS for ch in text):
        vi_score += 2
    if re.search(r"\b(the|bank|customers|implemented|launched|deployed)\b", lower):
        en_score += 2

    if vi_score >= 2 and en_score >= 2:
        return "mixed"
    if vi_score > en_score:
        return "vi"
    if en_score > vi_score:
        return "en"
    return "unknown"


def semantic_signature(text: str) -> set:
    """Return a compact cross-language signature for duplicate detection."""
    lower = strip_accents(text.lower())
    original_lower = text.lower()
    signature = set(re.findall(r"\b\d{4}\b", lower))

    if _ACTION_PATTERN.search(original_lower):
        signature.add("adoption_action")

    for concept, pattern in _TECH_PATTERNS.items():
        if re.search(pattern, original_lower, re.IGNORECASE):
            signature.add(concept)

    token_map = {
        "ngan": "bank",
        "hang": "bank",
        "bank": "bank",
        "khach": "customer",
        "customer": "customer",
        "customers": "customer",
        "ca": "individual",
        "nhan": "individual",
        "individual": "individual",
        "truc": "online",
        "tuyen": "online",
        "online": "online",
        "so": "digital",
        "digital": "digital",
    }
    for token in re.findall(r"\b[a-z0-9]+\b", lower):
        mapped = token_map.get(token)
        if mapped:
            signature.add(mapped)

    return signature


def signature_similarity(left: str, right: str) -> float:
    left_sig = semantic_signature(left)
    right_sig = semantic_signature(right)
    if not left_sig or not right_sig:
        return 0.0
    return len(left_sig & right_sig) / len(left_sig | right_sig)


def _evidence_score(seg: Segment) -> Tuple[int, int, int]:
    sig = semantic_signature(seg.text)
    action_score = 1 if "adoption_action" in sig else 0
    tech_score = sum(1 for item in sig if item in _TECH_PATTERNS)
    number_score = len(re.findall(r"\d+", seg.text))
    return action_score, tech_score, number_score


def _choose_canonical(current: Segment, candidate: Segment) -> Segment:
    current_score = _evidence_score(current)
    candidate_score = _evidence_score(candidate)
    if candidate_score > current_score:
        return candidate
    if candidate_score < current_score:
        return current
    if current.language != "vi" and candidate.language == "vi":
        return candidate
    if current.language == "vi" and candidate.language != "vi":
        return current
    return candidate if candidate.word_count > current.word_count else current

def _ordered_languages(languages: List[str]) -> List[str]:
    order = {"vi": 0, "en": 1, "mixed": 2, "unknown": 3}
    return sorted(set(languages), key=lambda lang: (order.get(lang, 99), lang))


def deduplicate_bilingual_segments(
    segments: List[Segment],
    similarity_threshold: float = 0.55,
) -> Tuple[List[Segment], List[Dict]]:
    """
    Keep one canonical paragraph when Vietnamese and English paragraphs express
    the same disclosure. This prevents bilingual annual reports from double
    counting one adoption event.
    """
    canonical: List[Segment] = []
    duplicate_rows: List[Dict] = []

    for seg in segments:
        seg.language = detect_language(seg.text)
        matched_index = None
        matched_similarity = 0.0

        for idx, existing in enumerate(canonical):
            language_pair = {existing.language, seg.language}
            if len(language_pair - {"vi", "en", "mixed", "unknown"}) > 0:
                continue
            similarity = signature_similarity(existing.text, seg.text)
            if similarity >= similarity_threshold:
                matched_index = idx
                matched_similarity = similarity
                break

        if matched_index is None:
            canonical.append(seg)
            continue

        existing = canonical[matched_index]
        chosen = _choose_canonical(existing, seg)
        duplicate = seg if chosen is existing else existing
        all_duplicate_ids = sorted(set(existing.duplicate_seg_ids + seg.duplicate_seg_ids + [duplicate.seg_id]))
        languages = _ordered_languages(
            existing.languages_in_cluster + seg.languages_in_cluster + [existing.language, seg.language]
        )

        chosen.duplicate_seg_ids = [sid for sid in all_duplicate_ids if sid != chosen.seg_id]
        chosen.languages_in_cluster = languages
        chosen.canonical_seg_id = chosen.seg_id
        chosen.duplicate_similarity = matched_similarity
        canonical[matched_index] = chosen

        duplicate_rows.append({
            "canonical_seg_id": chosen.seg_id,
            "duplicate_seg_id": duplicate.seg_id,
            "canonical_language": chosen.language,
            "duplicate_language": duplicate.language,
            "similarity": round(matched_similarity, 4),
            "reason": "bilingual_semantic_duplicate",
        })

    return canonical, duplicate_rows


def split_into_paragraphs(
    text: str,
    min_words: int = 15,
    max_words: int = 200,
) -> List[Tuple[str, int]]:
    """
    Split cleaned full-document text into paragraph-level segments.
    """
    raw_paragraphs = re.split(r"\n{2,}", text)
    results: List[Tuple[str, int]] = []
    char_pos = 0

    for para in raw_paragraphs:
        para = para.strip()
        offset = text.find(para, char_pos)
        char_pos = offset + len(para)

        word_count = len(para.split())
        if word_count < min_words:
            continue

        if word_count <= max_words:
            results.append((para, offset))
        else:
            sentences = _split_sentences(para)
            chunks = _sliding_window(sentences, max_words, overlap=1)
            for chunk in chunks:
                chunk_text = " ".join(chunk)
                if len(chunk_text.split()) >= min_words:
                    chunk_offset = text.find(chunk_text[:30], offset)
                    results.append((chunk_text, max(offset, chunk_offset)))

    return results


def _split_sentences(text: str) -> List[str]:
    pattern = re.compile(
        r"(?<=[.!?])\s+(?=[A-ZÁÀẢÃẠĂẮẶẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐĐ])"
    )
    parts = pattern.split(text)
    return [s.strip() for s in parts if s.strip()]


def _sliding_window(
    sentences: List[str],
    max_words: int,
    overlap: int = 1,
) -> List[List[str]]:
    chunks = []
    start = 0
    while start < len(sentences):
        current, word_count = [], 0
        i = start
        while i < len(sentences):
            wc = len(sentences[i].split())
            if word_count + wc > max_words and current:
                break
            current.append(sentences[i])
            word_count += wc
            i += 1
        if current:
            chunks.append(current)
        start = max(start + 1, i - overlap)
        if start >= len(sentences):
            break
    return chunks


def extract_with_pdfplumber(pdf_path: str) -> List[Tuple[str, int]]:
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pip install pdfplumber")

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                pages.append((text, i + 1))
    return pages


def extract_with_pypdf(pdf_path: str) -> List[Tuple[str, int]]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("pip install pypdf")

    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((text, i + 1))
    return pages


def preprocess_report(
    pdf_path: str,
    bank_id: str,
    year: int,
    min_words: int = 15,
    max_words: int = 200,
    backend: str = "auto",
    bilingual_dedupe_threshold: float = 0.55,
    fallback_backend: str = "pymupdf_fast",
    backend_timeout_seconds: int = 180,
    enable_ocr: bool = True,
    ocr_mode: str = "skip",
    ocr_output_dir: str = "data/processed/ocr",
    ocr_invalidate_digital_signatures: bool = False,
    ocr_jobs: int = 2,
    ocr_timeout_seconds: int = 1800,
    min_words_per_text_page: int = 10,
    min_text_page_share: float = 0.50,
    long_document_min_pages: int = 30,
    min_words_per_page_long_document: float = 80.0,
) -> BankReport:
    """Full preprocessing pipeline for one annual report PDF."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"[Preprocessor] Processing {path.name} ({bank_id}, {year})")

    preferred_backend = {
        "auto": "pymupdf4llm",
        "pymupdf4llm": "pymupdf4llm",
        "pymupdf_fast": "pymupdf_fast",
        "pdfplumber": "pdfplumber",
        "pypdf": "pypdf",
    }.get(backend, "pymupdf4llm")
    read_result = read_pdf_auto(
        pdf_path,
        DocumentReaderConfig(
            preferred_backend=preferred_backend,
            fallback_backend=fallback_backend,
            backend_timeout_seconds=backend_timeout_seconds,
            enable_ocr=enable_ocr,
            ocr_mode=ocr_mode,
            ocr_output_dir=ocr_output_dir,
            ocr_invalidate_digital_signatures=ocr_invalidate_digital_signatures,
            ocr_jobs=ocr_jobs,
            ocr_timeout_seconds=ocr_timeout_seconds,
            min_words_per_text_page=min_words_per_text_page,
            min_text_page_share=min_text_page_share,
            long_document_min_pages=long_document_min_pages,
            min_words_per_page_long_document=min_words_per_page_long_document,
        ),
    )
    raw_pages = [(page.text, page.page_num) for page in read_result.pages]

    logger.info(f"  Extracted {len(raw_pages)} pages")

    page_texts: List[Tuple[str, int]] = []
    for raw_text, page_num in raw_pages:
        cleaned = clean_text(raw_text)
        if len(cleaned.split()) >= 10:
            page_texts.append((cleaned, page_num))

    full_text = "\n\n".join(t for t, _ in page_texts)
    raw_total_words = read_result.raw_total_words or sum(
        len(raw_text.split()) for raw_text, _ in raw_pages
    )
    clean_total_words = len(full_text.split())
    para_chunks = split_into_paragraphs(full_text, min_words, max_words)

    page_offset_map: List[Tuple[int, int]] = []
    cum = 0
    for cleaned, page_num in page_texts:
        cum += len(cleaned) + 2
        page_offset_map.append((cum, page_num))

    def page_for_offset(offset: int) -> int:
        for end, pnum in page_offset_map:
            if offset < end:
                return pnum
        return page_offset_map[-1][1] if page_offset_map else 1

    raw_segments: List[Segment] = []
    for seg_id, (seg_text, char_offset) in enumerate(para_chunks):
        raw_segments.append(
            Segment(
                seg_id=seg_id,
                text=seg_text,
                page_num=page_for_offset(char_offset),
                char_offset=char_offset,
            )
        )

    canonical_segments, duplicates = deduplicate_bilingual_segments(
        raw_segments,
        similarity_threshold=bilingual_dedupe_threshold,
    )
    canonical_clean_words = sum(seg.word_count for seg in canonical_segments)

    report = BankReport(
        bank_id=bank_id,
        year=year,
        pdf_path=str(pdf_path),
        segments=canonical_segments,
        clean_total_words=clean_total_words,
        raw_total_words=raw_total_words,
        canonical_clean_words=canonical_clean_words,
        duplicate_segments=duplicates,
        metadata={
            "num_pages": read_result.num_pages or len(raw_pages),
            "pdf_backend": read_result.backend,
            "extraction_status": read_result.status,
            "ocr_used": read_result.ocr_used,
            "ocr_engine": read_result.ocr_engine,
            "ocr_pdf_path": read_result.ocr_pdf_path,
            "scan_detected": read_result.scan_detected,
            "text_coverage": read_result.text_coverage,
            "text_pages_ge10": read_result.text_pages_ge10,
            "text_pages_ge50": read_result.text_pages_ge50,
            "text_page_share_ge10": read_result.text_page_share_ge10,
            "words_per_physical_page": read_result.words_per_physical_page,
            "median_words_per_page": read_result.median_words_per_page,
            "page_quality": read_result.page_quality,
            "garbled_ratio": read_result.garbled_ratio,
            "extraction_warnings": read_result.extraction_warnings,
            "n_raw_segments": len(raw_segments),
            "n_canonical_segments": len(canonical_segments),
            "n_bilingual_duplicates": len(duplicates),
        },
    )

    logger.info(
        f"  Segmented -> {len(raw_segments)} paragraphs | "
        f"canonical={len(canonical_segments)} | "
        f"canonical words={report.canonical_clean_words:,}"
    )
    return report


def preprocess_batch(
    file_map: dict,
    min_words: int = 15,
    max_words: int = 200,
    bilingual_dedupe_threshold: float = 0.55,
    fallback_backend: str = "pymupdf_fast",
    backend_timeout_seconds: int = 180,
    enable_ocr: bool = True,
    ocr_mode: str = "skip",
    ocr_output_dir: str = "data/processed/ocr",
    ocr_invalidate_digital_signatures: bool = False,
    ocr_jobs: int = 2,
    ocr_timeout_seconds: int = 1800,
    min_words_per_text_page: int = 10,
    min_text_page_share: float = 0.50,
    long_document_min_pages: int = 30,
    min_words_per_page_long_document: float = 80.0,
    failures: Optional[List[Dict[str, object]]] = None,
) -> List[BankReport]:
    """Process multiple reports."""
    reports = []
    for (bank_id, year), pdf_path in file_map.items():
        try:
            report = preprocess_report(
                pdf_path,
                bank_id,
                year,
                min_words,
                max_words,
                bilingual_dedupe_threshold=bilingual_dedupe_threshold,
                fallback_backend=fallback_backend,
                backend_timeout_seconds=backend_timeout_seconds,
                enable_ocr=enable_ocr,
                ocr_mode=ocr_mode,
                ocr_output_dir=ocr_output_dir,
                ocr_invalidate_digital_signatures=ocr_invalidate_digital_signatures,
                ocr_jobs=ocr_jobs,
                ocr_timeout_seconds=ocr_timeout_seconds,
                min_words_per_text_page=min_words_per_text_page,
                min_text_page_share=min_text_page_share,
                long_document_min_pages=long_document_min_pages,
                min_words_per_page_long_document=min_words_per_page_long_document,
            )
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed {bank_id} {year}: {e}")
            if failures is not None:
                failures.append({
                    "bank_id": bank_id,
                    "year": int(year),
                    "pdf_path": str(pdf_path),
                    "stage": "preprocess",
                    "error": str(e),
                })
    return reports
