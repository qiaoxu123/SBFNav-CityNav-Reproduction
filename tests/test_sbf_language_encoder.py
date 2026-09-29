import numpy as np
import torch

from sbfnav.models.language_encoder import FrozenSiglipBackbone
from sbfnav.runtime import TextFeatureCache


def test_offline_siglip_loads_frozen_token_features():
    encoder = FrozenSiglipBackbone(local_files_only=True, freeze=True)
    features = encoder.encode_text(["the red car next to the named building"], "cpu")
    assert features.tokens.shape == (1, 64, 768)
    assert features.pooled.shape == (1, 768)
    assert features.attention_mask.shape == (1, 64)
    assert features.attention_mask.dtype == torch.bool
    assert features.attention_mask.any()
    assert not features.attention_mask.all()
    assert torch.isfinite(features.tokens).all()
    assert not any(parameter.requires_grad for parameter in encoder.model.parameters())
    assert encoder.model.training is False
    encoder.train()
    assert encoder.model.training is False


def test_vectorized_float_image_preprocessing_matches_official_processor():
    encoder = FrozenSiglipBackbone(local_files_only=True, freeze=True)
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(2, 224, 224, 3), dtype=np.uint8)
    official = encoder.preprocess_images(list(images), "cpu")
    vectorized = encoder.preprocess_float_images(
        images.astype(np.float32) / 255.0, "cpu"
    )
    torch.testing.assert_close(vectorized, official, atol=1e-6, rtol=0)


def test_frozen_text_precompute_deduplicates_and_reuses_features():
    encoder = FrozenSiglipBackbone(local_files_only=True, freeze=True)
    cache = TextFeatureCache(encoder, torch.device("cpu"))
    cache.precompute(["a landmark", "a landmark", "another place"], batch_size=1)
    assert set(cache._cache) == {"a landmark", "another place"}
    before = {key: value[0].data_ptr() for key, value in cache._cache.items()}
    features = cache.get(["another place", "a landmark"])
    assert features.tokens.shape == (2, 64, 768)
    assert before == {key: value[0].data_ptr() for key, value in cache._cache.items()}
