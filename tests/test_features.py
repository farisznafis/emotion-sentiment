import numpy as np

from src.config import (
    EXPECTED_FRAME_COUNT,
    FEATURE_COUNT,
    FEATURE_SAMPLE_RATE,
    TARGET_SAMPLES,
)
from src.features import extract_features


def _make_test_audio() -> np.ndarray:
    return (
        np.random.default_rng(
            seed=42
        )
        .normal(
            size=TARGET_SAMPLES
        )
        .astype(np.float32)
    )


def test_extract_features_returns_expected_shape():
    features = extract_features(
        _make_test_audio(),
        FEATURE_SAMPLE_RATE,
    )

    assert features.shape == (
        1,
        EXPECTED_FRAME_COUNT,
        FEATURE_COUNT,
    )


def test_extract_features_are_finite():
    features = extract_features(
        _make_test_audio(),
        FEATURE_SAMPLE_RATE,
    )

    assert np.all(
        np.isfinite(features)
    )