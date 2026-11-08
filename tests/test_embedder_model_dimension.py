import numpy as np

from fintech_adoption.embedder import FinTechEmbedder, _embedding_dimension


class ModelWithNewDimensionApi:
    def get_embedding_dimension(self):
        return 768


class ModelWithLegacyDimensionApi:
    def get_sentence_embedding_dimension(self):
        return 384


def test_embedding_dimension_prefers_current_sentence_transformers_api():
    assert _embedding_dimension(ModelWithNewDimensionApi()) == 768


def test_embedding_dimension_supports_legacy_sentence_transformers_api():
    assert _embedding_dimension(ModelWithLegacyDimensionApi()) == 384


def test_empty_embeddings_match_cached_seed_dimension_without_loading_model():
    embedder = FinTechEmbedder()
    embedder.seed_embeddings = np.ones((3, 384), dtype=np.float32)
    assert embedder.encode([]).shape == (0, 384)
    assert embedder._model is None
