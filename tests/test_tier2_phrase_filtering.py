from fintech_adoption.tier2_scorer import extract_ngrams_basic, is_valid_phrase


def test_basic_ngram_extractor_filters_noise():
    text = "Ngân hàng đã triển khai eKYC cho khách hàng năm 2023 và đạt 2 triệu users."

    phrases = extract_ngrams_basic(text, 1, 4)

    assert "eKYC" in phrases
    assert "triển khai eKYC" in phrases
    assert "2023" not in phrases
    assert "và" not in phrases


def test_is_valid_phrase_rejects_digit_heavy_and_stopword_only_phrases():
    assert is_valid_phrase("eKYC") is True
    assert is_valid_phrase("triển khai eKYC") is True
    assert is_valid_phrase("2023") is False
    assert is_valid_phrase("và của") is False
