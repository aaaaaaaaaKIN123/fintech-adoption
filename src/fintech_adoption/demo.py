"""
demo.py — End-to-end demo of the FTA pipeline using synthetic text.

The default uses hashed text features and mock adjudication, entirely offline.
Use --semantic to run the configured sentence-transformer model instead.
All inputs and scores are synthetic examples, unsuitable for research claims.

Run:
  python demo.py
"""

import argparse
import logging
import os
import sys
import json

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("demo")



# SYNTHETIC ANNUAL REPORT TEXT


SYNTHETIC_REPORT = """
BÁO CÁO THƯỜNG NIÊN 2023 — NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN VÍ DỤ

Mục tiêu chiến lược chuyển đổi số

Trong năm 2023, Ngân hàng đã triển khai thành công hệ thống eKYC (định danh khách hàng điện tử)
cho toàn bộ quy trình mở tài khoản trực tuyến. Khách hàng có thể mở tài khoản chỉ trong 5 phút
thông qua ứng dụng di động mà không cần đến chi nhánh. Tính đến cuối năm 2023, hơn 2,3 triệu
khách hàng mới đã được onboarding qua kênh số, tăng 145% so với năm 2022.

Ứng dụng trí tuệ nhân tạo trong hoạt động tín dụng

Ngân hàng đã tích hợp mô hình học máy (machine learning) vào quy trình chấm điểm tín dụng.
Mô hình AI sử dụng hơn 200 biến dữ liệu, bao gồm dữ liệu hành vi giao dịch và lịch sử tín dụng,
để đưa ra quyết định phê duyệt vay tự động trong vòng 30 giây. Tỷ lệ nợ xấu trong danh mục vay
được phê duyệt tự động thấp hơn 23% so với quy trình thủ công truyền thống.

Hệ thống phòng chống gian lận dựa trên AI

Năm 2023, Ngân hàng ra mắt hệ thống phát hiện gian lận theo thời gian thực sử dụng thuật toán
phát hiện bất thường (anomaly detection). Hệ thống xử lý hơn 50 triệu giao dịch mỗi ngày và
đã ngăn chặn thiệt hại ước tính 320 tỷ đồng từ các hành vi gian lận.

Thanh toán số và hệ sinh thái FinTech

Ngân hàng đã ra mắt tính năng thanh toán QR code thế hệ mới tương thích với tiêu chuẩn NAPAS
VietQR. Trong quý 4/2023, số lượng giao dịch qua QR code tăng 280% so với cùng kỳ năm trước.
Ngân hàng cũng tích hợp ví điện tử với 12 đối tác FinTech, mở rộng hệ sinh thái thanh toán số.

Điện toán đám mây và hạ tầng kỹ thuật số

Trong năm 2023, Ngân hàng hoàn thành giai đoạn 1 của lộ trình chuyển đổi đám mây (cloud migration),
di chuyển 60% hệ thống phi cốt lõi lên môi trường hybrid cloud. Hạ tầng mới cho phép tăng quy mô
xử lý giao dịch lên 3 lần mà không cần đầu tư thêm phần cứng.

Rủi ro và thách thức trong quá trình số hóa

Quá trình chuyển đổi số cũng đặt ra những thách thức đáng kể về an ninh mạng và bảo vệ dữ liệu
khách hàng. Ngành ngân hàng nói chung đang đối mặt với nguy cơ tấn công mạng ngày càng tinh vi.
Ngân hàng sẽ tiếp tục đầu tư vào các giải pháp bảo mật trong các năm tiếp theo.

Kế hoạch tương lai (2024-2026)

Ngân hàng dự kiến sẽ triển khai open banking API vào năm 2024, cho phép đối tác bên thứ ba
tích hợp dịch vụ tài chính. Chúng tôi cũng đang nghiên cứu ứng dụng công nghệ blockchain
trong các giao dịch liên ngân hàng và dự kiến thí điểm vào năm 2025.

Dịch vụ ngân hàng số cá nhân hóa

Ứng dụng ngân hàng di động của chúng tôi hiện sử dụng AI để cá nhân hóa giao diện và đề xuất
sản phẩm phù hợp cho từng khách hàng. Hệ thống recommendation engine phân tích hành vi sử dụng
để đưa ra gợi ý sản phẩm có tỷ lệ chuyển đổi cao hơn 35% so với phương pháp truyền thống.

Tự động hóa quy trình nội bộ (RPA)

Ngân hàng đã triển khai 47 robot phần mềm (RPA) trong các quy trình back-office, bao gồm
xử lý chứng từ, đối soát giao dịch và báo cáo giám sát. Các robot này xử lý hơn 1,2 triệu
giao dịch mỗi tháng với độ chính xác 99.97%, giảm 68% chi phí vận hành liên quan.

Kết quả kinh doanh tổng quan

Tổng tài sản đạt 485.000 tỷ đồng, tăng 12% so với năm 2022. Lợi nhuận trước thuế đạt 8.500 tỷ đồng.
Tỷ lệ CAR đạt 13,2%, cao hơn mức tối thiểu theo quy định Basel II. Chi phí hoạt động trên thu nhập
(CIR) giảm còn 38%, một phần nhờ hiệu quả từ chuyển đổi số và tự động hóa.
"""



# MOCK PDF CREATION


def create_mock_pdf(text: str, output_path: str) -> str:
    """Create a minimal PDF containing the demo text."""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        import textwrap

        c = canvas.Canvas(output_path, pagesize=A4)
        width, height = A4
        y = height - 50
        c.setFont("Helvetica", 9)

        for line in text.strip().split("\n"):
            wrapped = textwrap.wrap(line, width=100)
            for wline in wrapped:
                if y < 50:
                    c.showPage()
                    y = height - 50
                    c.setFont("Helvetica", 9)
                c.drawString(40, y, wline)
                y -= 14
        c.save()
        return output_path
    except ImportError:
        # If reportlab not available, create a text file and wrap as fake pdf path
        txt_path = output_path.replace(".pdf", ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        return txt_path



# DEMO RUNNER (bypasses PDF extraction for text-only demo)


def run_demo_from_text(
    text: str, bank_id: str = "DEMO_BANK", year: int = 2023,
    semantic: bool = False, output_dir: str | None = None,
):
    """
    Run the pipeline directly from raw text (no PDF extraction needed).
    Simulates what the pipeline does after Phase 1.
    """
    logger.info("\n" + "="*60)
    logger.info("  FINTECH ADOPTION PIPELINE — DEMO MODE")
    logger.info("="*60)

    # Import pipeline modules
    from .config import PipelineConfig
    from .seed_concepts import get_concept_texts, summarize
    from .preprocessor import (
        split_into_paragraphs, clean_text,
        BankReport, Segment
    )
    from .embedder import FinTechEmbedder
    from .tier1_scorer import score_tier1
    from .tier2_scorer import Tier2Scorer
    from .llm_adjudicator import LLMAdjudicator
    from .aggregator import aggregate_report

    cfg = PipelineConfig()
    cfg.run_llm_adjudication = True
    cfg.tier1.top_k = 15  # Small for demo
    cfg.tier1.alpha = 0.65
    if output_dir is not None:
        cfg.output_dir = output_dir

    # Phase 0: Seeds
    logger.info("\n[Phase 0] Initialising seed concepts ...")
    summarize()
    class OfflineDemoEmbedder(FinTechEmbedder):
        """Deterministic text features for demonstrating the pipeline offline."""

        def encode(self, texts, show_progress=False):
            import numpy as np
            from sklearn.feature_extraction.text import HashingVectorizer

            if not texts:
                return np.zeros((0, 256), dtype=np.float32)
            vectorizer = HashingVectorizer(n_features=256, alternate_sign=False)
            return vectorizer.transform(texts).toarray().astype(np.float32)

    embedder_cls = FinTechEmbedder if semantic else OfflineDemoEmbedder
    embedder = embedder_cls(
        model_name=cfg.embedding.model_name,
        device="cpu",
        batch_size=32,
        cache_dir=cfg.embedding.cache_dir,
    )

    seed_texts = get_concept_texts()
    logger.info(f"  Encoding {len(seed_texts)} seed concepts ...")
    embedder.encode_seeds(seed_texts)

    # Phase 1: Segment synthetic text
    logger.info("\n[Phase 1] Segmenting text ...")
    cleaned = clean_text(text)
    para_chunks = split_into_paragraphs(
        cleaned,
        min_words=cfg.segmentation.min_segment_words,
        max_words=cfg.segmentation.max_segment_words,
    )

    segments = [
        Segment(seg_id=i, text=seg_text, page_num=1, char_offset=offset)
        for i, (seg_text, offset) in enumerate(para_chunks)
    ]
    total_words = sum(s.word_count for s in segments)

    report = BankReport(
        bank_id=bank_id,
        year=year,
        pdf_path="[synthetic]",
        segments=segments,
        total_words=total_words,
        metadata={"num_pages": 1},
    )
    logger.info(f"  Segments: {len(segments)} | Words: {total_words:,}")

    # Phase 2: Embed segments
    logger.info("\n[Phase 2] Embedding segments ...")
    seg_texts = [s.text for s in segments]
    seg_embs = embedder.encode_segments(seg_texts)

    # Phase 3: Tier-1
    logger.info("\n[Phase 3] Tier-1 scoring ...")
    top_k_segs, top_k_embs = score_tier1(report, seg_embs, embedder, cfg.tier1)

    logger.info(f"\n  Top-{len(top_k_segs)} segments by S^(1):")
    for i, seg in enumerate(top_k_segs[:5]):
        logger.info(f"    [{i+1}] S1={seg.S1_j:.3f} R={seg.R_j:.3f} C={seg.C_j:.3f} | {seg.text[:80]}...")

    # Phase 4: Tier-2
    logger.info("\n[Phase 4] Tier-2 phrase scoring ...")
    t2_scorer = Tier2Scorer(embedder=embedder, cfg=cfg.tier2)
    t2_scorer.score_segments(top_k_segs, top_n=3)

    logger.info("\n  Top phrases per segment:")
    for seg in top_k_segs[:5]:
        logger.info(f"    S2={seg.S2_j:.4f} | Phrases: {seg.key_phrases}")

    # Phase 5: LLM adjudication (MOCK)
    logger.info("\n[Phase 5] LLM adjudication (mock mode) ...")
    adj = LLMAdjudicator(cfg=cfg.llm, mock=True, rate_limit_delay=0.0)
    adj.adjudicate_segments(top_k_segs)

    label_summary = {}
    for seg in top_k_segs:
        label_summary[seg.llm_label] = label_summary.get(seg.llm_label, 0) + 1
    logger.info(f"  Label distribution: {label_summary}")

    # Phase 6: Aggregation
    logger.info("\n[Phase 6] Computing FTA index ...")
    record = aggregate_report(report, top_k_segs, cfg.aggregation)

    # Final Output
    print("\n" + "="*60)
    print(f"  FINAL RESULT: {bank_id} {year}")
    print("="*60)
    print(f"  FTA_it (continuous)  = {record.FTA_it:.6f}")
    print(f"  FTA_binary (0/1)     = {record.FTA_binary}")
    print(f"  Segments ADOPTED     = {record.n_segments_adopted}")
    print(f"  Segments DISCUSSED   = {record.n_segments_discussed}")
    print(f"  Total words (L_it)   = {record.total_words:,}")
    print(f"  Avg S^(1) (Tier-1)   = {record.avg_S1:.4f}")
    print(f"  Avg S^(2) (Tier-2)   = {record.avg_S2:.4f}")
    print(f"  Avg P_LLM            = {record.avg_P_llm:.4f}")
    print("\n  Top FinTech phrases detected:")
    for phrase in record.top_key_phrases[:10]:
        print(f"    • {phrase}")
    print("="*60)

    # Save output
    os.makedirs(cfg.output_dir, exist_ok=True)
    result_path = os.path.join(cfg.output_dir, f"{bank_id}_{year}_demo_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        from dataclasses import asdict
        json.dump({"synthetic": True, "embedding_mode": "semantic" if semantic else "hashing",
                   "llm_mode": "mock", "record": asdict(record)},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  Synthetic result saved: {result_path}")

    return record



# MAIN


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Run a synthetic FinTech adoption demo.")
    parser.add_argument("--semantic", action="store_true", help="Use SBERT; may download model weights.")
    parser.add_argument("--output", default=None, help="Directory for synthetic output JSON.")
    args = parser.parse_args()
    run_demo_from_text(
        text=SYNTHETIC_REPORT,
        bank_id="DEMO_BANK",
        year=2023,
        semantic=args.semantic,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
