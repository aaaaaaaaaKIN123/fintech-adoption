"""
seed_concepts.py — Structured seed concept pool for FinTech adoption measurement.

Architecture
────────────
  Prior seed pool  →  Concept groups  →  Validated seed concept set
  (V^(0), N=50-100)   (C = {c_m})        (used in all three branches)

Each concept c_m has:
  - concept_id   : unique snake_case identifier
  - domain       : thematic category for grouping
  - aliases      : List[str] of English/Vietnamese surface forms

This file defines the FULL seed concept set 𝒞 = {c_1, ..., c_M}.
Extend this list based on domain validation and embedding discovery.
"""

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SeedConcept:
    concept_id: str
    domain: str
    aliases: List[str]
    description: str = ""



# DOMAIN 1 — DIGITAL IDENTITY & CUSTOMER ONBOARDING

DOMAIN_1 = [
    SeedConcept(
        "ekyc_identity",
        "digital_identity",
        ["eKYC", "electronic KYC", "digital KYC", "know your customer digital",
         "định danh điện tử", "xác minh danh tính điện tử", "định danh khách hàng điện tử",
         "KYC điện tử", "nhận diện khách hàng trực tuyến", "eKYC sinh trắc học"],
        "Electronic Know-Your-Customer identity verification",
    ),
    SeedConcept(
        "biometric_auth",
        "digital_identity",
        ["biometric authentication", "facial recognition", "fingerprint recognition",
         "voice recognition", "iris scan", "liveness detection",
         "xác thực sinh trắc học", "nhận diện khuôn mặt", "nhận diện vân tay",
         "nhận diện giọng nói", "xác thực khuôn mặt", "xác thực dấu vân tay"],
        "Biometric-based user authentication",
    ),
    SeedConcept(
        "digital_onboarding",
        "digital_identity",
        ["digital onboarding", "online account opening", "remote onboarding",
         "paperless onboarding", "straight-through processing",
         "mở tài khoản trực tuyến", "mở tài khoản số", "đăng ký trực tuyến",
         "onboarding kỹ thuật số", "tiếp nhận khách hàng số"],
        "Digital customer acquisition and onboarding process",
    ),
]


# DOMAIN 2 — ARTIFICIAL INTELLIGENCE & MACHINE LEARNING

DOMAIN_2 = [
    SeedConcept(
        "ai_analytics",
        "ai_ml",
        ["artificial intelligence", "AI", "machine learning", "ML", "deep learning",
         "neural network", "AI-powered", "AI-driven", "AI application",
         "trí tuệ nhân tạo", "học máy", "học sâu", "mạng nơ-ron",
         "ứng dụng AI", "giải pháp AI", "AI trong ngân hàng"],
        "AI/ML applications in banking",
    ),
    SeedConcept(
        "nlp_chatbot",
        "ai_ml",
        ["natural language processing", "NLP", "chatbot", "virtual assistant",
         "conversational AI", "large language model", "LLM", "GPT", "generative AI",
         "xử lý ngôn ngữ tự nhiên", "trợ lý ảo", "chatbot ngân hàng",
         "tư vấn ảo", "robot tư vấn", "trợ lý số"],
        "NLP-based customer interaction tools",
    ),
    SeedConcept(
        "credit_scoring_ai",
        "ai_ml",
        ["AI credit scoring", "ML credit model", "alternative data credit",
         "automated underwriting", "credit risk AI", "behavioural scoring",
         "chấm điểm tín dụng AI", "mô hình tín dụng học máy",
         "phân tích tín dụng tự động", "xếp hạng tín dụng số",
         "dữ liệu thay thế tín dụng"],
        "AI-based credit assessment",
    ),
    SeedConcept(
        "fraud_detection_ai",
        "ai_ml",
        ["fraud detection", "anti-fraud AI", "anomaly detection", "transaction monitoring AI",
         "real-time fraud prevention", "AML machine learning",
         "intelligent risk control", "smart risk control",
         "phát hiện gian lận", "chống gian lận AI", "phát hiện bất thường",
         "giám sát giao dịch AI", "phòng chống rửa tiền AI",
         "kiểm soát rủi ro thông minh", "kiểm soát rủi ro AI"],
        "AI-driven fraud and AML detection",
    ),
    SeedConcept(
        "robo_advisory",
        "ai_ml",
        ["robo-advisor", "robo advisory", "automated investment", "algo trading",
         "algorithmic investment", "wealth-tech",
         "tư vấn đầu tư tự động", "robo-advisor", "đầu tư thuật toán",
         "quản lý tài sản tự động"],
        "Automated investment and advisory",
    ),
]


# DOMAIN 3 — DIGITAL PAYMENTS & TRANSACTIONS

DOMAIN_3 = [
    SeedConcept(
        "digital_payment",
        "payment",
        ["digital payment", "e-payment", "electronic payment", "contactless payment",
         "QR payment", "NFC payment", "mobile payment", "instant payment",
         "thanh toán số", "thanh toán điện tử", "thanh toán không tiếp xúc",
         "thanh toán QR", "thanh toán di động", "thanh toán nhanh"],
        "Digital and electronic payment systems",
    ),
    SeedConcept(
        "mobile_banking",
        "payment",
        ["mobile banking", "mobile app banking", "banking app", "mobile wallet",
         "digital wallet", "e-wallet",
         "ngân hàng di động", "ứng dụng ngân hàng", "ví điện tử",
         "ví số", "ngân hàng trên điện thoại", "app ngân hàng"],
        "Mobile banking and wallet applications",
    ),
    SeedConcept(
        "open_banking_api",
        "payment",
        ["open banking", "API banking", "banking API", "open API", "PSD2",
         "third-party provider", "account aggregation", "bank-as-a-service",
         "ngân hàng mở", "API ngân hàng", "nền tảng mở", "kết nối bên thứ ba"],
        "Open banking and API integration",
    ),
    SeedConcept(
        "real_time_payment",
        "payment",
        ["real-time payment", "instant payment", "24/7 payment", "fast payment",
         "NAPAS", "VNPay", "interbank instant transfer",
         "thanh toán tức thời", "chuyển tiền nhanh 24/7", "thanh toán liên ngân hàng tức thời"],
        "Real-time interbank payment infrastructure",
    ),
]


# DOMAIN 4 — CLOUD COMPUTING & INFRASTRUCTURE

DOMAIN_4 = [
    SeedConcept(
        "cloud_computing",
        "infrastructure",
        ["cloud computing", "cloud infrastructure", "cloud migration", "SaaS",
         "IaaS", "PaaS", "hybrid cloud", "multi-cloud", "cloud-native",
         "financial cloud", "private cloud", "cloud services", "cloud platform",
         "điện toán đám mây", "hạ tầng đám mây", "chuyển đổi đám mây",
         "dịch vụ đám mây", "nền tảng đám mây", "đám mây tài chính",
         "đám mây riêng", "dịch vụ cloud", "nền tảng cloud"],
        "Cloud computing adoption",
    ),
    SeedConcept(
        "core_banking_digital",
        "infrastructure",
        ["core banking transformation", "digital core banking", "banking platform modernization",
         "legacy modernization", "microservices banking", "API-first architecture",
         "chuyển đổi core banking", "hiện đại hóa hệ thống lõi",
         "nền tảng ngân hàng số", "kiến trúc vi dịch vụ"],
        "Core banking system modernization",
    ),
    SeedConcept(
        "cybersecurity_digital",
        "infrastructure",
        ["cybersecurity", "information security", "data security", "cyber resilience",
         "zero trust", "endpoint security", "SIEM", "SOC",
         "quantum communication", "quantum cryptography",
         "an ninh mạng", "bảo mật thông tin", "bảo mật dữ liệu",
         "an toàn thông tin số", "trung tâm điều hành an ninh mạng",
         "truyền thông lượng tử", "mật mã lượng tử"],
        "Cybersecurity in digital banking",
    ),
    SeedConcept(
        "data_infrastructure",
        "infrastructure",
        ["big data", "data lake", "data warehouse", "data platform",
         "real-time analytics", "data-driven", "data governance",
         "big data analytics", "big data services", "big data technology",
         "big data models", "big data mining", "data warehousing",
         "dữ liệu lớn", "hồ dữ liệu", "kho dữ liệu", "nền tảng dữ liệu",
         "phân tích dữ liệu thời gian thực", "quản trị dữ liệu"],
        "Data infrastructure and analytics platforms",
    ),
]


# DOMAIN 5 — BLOCKCHAIN & DISTRIBUTED LEDGER

DOMAIN_5 = [
    SeedConcept(
        "blockchain_dlt",
        "blockchain",
        ["blockchain", "distributed ledger", "DLT", "smart contract",
         "tokenization", "digital asset", "stablecoin", "CBDC",
         "chuỗi khối", "sổ cái phân tán", "hợp đồng thông minh",
         "tài sản số", "tiền kỹ thuật số ngân hàng trung ương"],
        "Blockchain and distributed ledger technology",
    ),
    SeedConcept(
        "cross_border_payment",
        "blockchain",
        ["cross-border payment", "international remittance", "SWIFT alternative",
         "correspondent banking digital", "FX automation",
         "thanh toán xuyên biên giới", "chuyển tiền quốc tế", "kiều hối điện tử",
         "thanh toán ngoại tệ tự động"],
        "Digital cross-border payment solutions",
    ),
]


# DOMAIN 6 — DIGITAL LENDING & FINANCIAL INCLUSION

DOMAIN_6 = [
    SeedConcept(
        "digital_lending",
        "lending",
        ["digital lending", "online lending", "automated loan", "instant loan",
         "BNPL", "buy now pay later", "P2P lending",
         "cho vay số", "cho vay trực tuyến", "vay tự động",
         "vay nhanh số", "phê duyệt vay tự động"],
        "Digital and automated lending products",
    ),
    SeedConcept(
        "financial_inclusion",
        "lending",
        ["financial inclusion", "unbanked", "underbanked", "micro-finance digital",
         "agent banking", "branchless banking",
         "tài chính toàn diện", "tài chính bao trùm", "ngân hàng không chi nhánh",
         "phổ cập tài chính số"],
        "Digital financial inclusion initiatives",
    ),
    SeedConcept(
        "regtech_compliance",
        "lending",
        ["RegTech", "regulatory technology", "compliance automation",
         "digital KYC compliance", "AML automation", "supervisory technology",
         "công nghệ tuân thủ", "tự động hóa tuân thủ", "giám sát tự động",
         "báo cáo giám sát điện tử"],
        "Regulatory technology and compliance automation",
    ),
]


# DOMAIN 7 — DIGITAL CHANNELS & CUSTOMER EXPERIENCE

DOMAIN_7 = [
    SeedConcept(
        "internet_banking",
        "channels",
        ["internet banking", "online banking", "digital banking platform",
         "omnichannel banking", "digital branch",
         "ngân hàng trực tuyến", "ngân hàng số", "nền tảng ngân hàng số",
         "đa kênh số", "chi nhánh số"],
        "Online banking platforms and omnichannel",
    ),
    SeedConcept(
        "personalization_cx",
        "channels",
        ["personalization", "customer experience digital", "hyper-personalization",
         "recommendation engine", "next-best-offer", "customer journey digital",
         "digital marketing", "marketing automation",
         "cá nhân hóa", "trải nghiệm khách hàng số", "gợi ý sản phẩm cá nhân",
         "hành trình khách hàng số", "marketing số", "tiếp thị số"],
        "Digital personalization and customer experience",
    ),
    SeedConcept(
        "digital_wealth_insurance",
        "channels",
        ["digital wealth management", "bancassurance digital", "insurtech",
         "digital investment platform", "e-insurance",
         "quản lý tài sản số", "bảo hiểm số", "bancassurance trực tuyến",
         "nền tảng đầu tư số"],
        "Digital wealth and insurance distribution",
    ),
]


# DOMAIN 8 — STRATEGIC & ORGANIZATIONAL DIGITAL TRANSFORMATION

DOMAIN_8 = [
    SeedConcept(
        "digital_transformation_strategy",
        "strategy",
        ["digital transformation", "digitalization strategy", "digital strategy",
         "FinTech partnership", "technology investment", "innovation lab",
         "chuyển đổi số", "chiến lược số hóa", "chiến lược kỹ thuật số",
         "hợp tác FinTech", "đầu tư công nghệ", "trung tâm đổi mới sáng tạo"],
        "Overall digital transformation strategy",
    ),
    SeedConcept(
        "fintech_ecosystem",
        "strategy",
        ["FinTech ecosystem", "startup collaboration", "technology partnership",
         "digital ecosystem", "platform economy", "embedded finance",
         "hệ sinh thái FinTech", "hợp tác công nghệ", "kinh tế nền tảng",
         "tài chính nhúng", "hệ sinh thái số"],
        "FinTech ecosystem and partnerships",
    ),
    SeedConcept(
        "automation_rpa",
        "strategy",
        ["robotic process automation", "RPA", "workflow automation",
         "intelligent automation", "process digitization", "straight-through processing",
         "robot", "software robot", "process robot",
         "tự động hóa quy trình", "RPA", "tự động hóa thông minh",
         "số hóa quy trình nội bộ", "robot phần mềm", "robot quy trình"],
        "Process automation and RPA",
    ),
    SeedConcept(
        "iot_digital_banking",
        "strategy",
        ["Internet of Things", "IoT banking", "wearable payment", "smart ATM",
         "connected devices banking", "embedded banking IoT",
         "Internet vạn vật", "thanh toán thiết bị đeo", "ATM thông minh",
         "ngân hàng kết nối thiết bị"],
        "IoT and connected device banking",
    ),
]


# MASTER SEED CONCEPT SET  𝒞 = {c_1, ..., c_M}

ALL_SEED_CONCEPTS: List[SeedConcept] = (
    DOMAIN_1 + DOMAIN_2 + DOMAIN_3 +
    DOMAIN_4 + DOMAIN_5 + DOMAIN_6 +
    DOMAIN_7 + DOMAIN_8
)

# Quick lookup: concept_id → SeedConcept
CONCEPT_MAP: Dict[str, SeedConcept] = {
    c.concept_id: c for c in ALL_SEED_CONCEPTS
}

# Flat list of all alias strings (used for keyword-matching branch)
ALL_ALIASES: List[str] = [
    alias
    for concept in ALL_SEED_CONCEPTS
    for alias in concept.aliases
]


def get_concept_texts() -> List[str]:
    """
    Return one representative text per concept for embedding.
    Uses the first English alias as the primary representation;
    all aliases are concatenated for richer coverage.
    """
    return [" | ".join(c.aliases) for c in ALL_SEED_CONCEPTS]


def get_concept_ids() -> List[str]:
    return [c.concept_id for c in ALL_SEED_CONCEPTS]


def summarize() -> None:
    domains = {}
    for c in ALL_SEED_CONCEPTS:
        domains.setdefault(c.domain, []).append(c.concept_id)
    print(f"\n{'='*60}")
    print(f"  FINTECH SEED CONCEPT POOL  |  M = {len(ALL_SEED_CONCEPTS)} concepts")
    print(f"  Total aliases (prior pool): {len(ALL_ALIASES)}")
    print(f"{'='*60}")
    for domain, ids in domains.items():
        print(f"  [{domain}]  {len(ids)} concepts: {', '.join(ids)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    summarize()
