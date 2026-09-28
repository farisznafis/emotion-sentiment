import numpy as np

from src.config import N_MFCC, TARGET_SAMPLES
from src.features import extract_features


def test_extract_features_returns_expected_shape():
    sample_rate = 22_050

    audio = np.random.default_rng(
        seed=42
    ).normal(
        size=TARGET_SAMPLES
    ).astype(
        np.float32
    )

    features = extract_features(
        audio,
        sample_rate,
    )

    expected_feature_count = (
        N_MFCC + 2
    )

    assert features.ndim == 3

    assert features.shape[0] == 1

    assert (
        features.shape[2]
        == expected_feature_count
    )


def test_extract_features_are_finite():
    sample_rate = 22_050

    audio = np.random.default_rng(
        seed=42
    ).normal(
        size=TARGET_SAMPLES
    ).astype(
        np.float32
    )

    features = extract_features(
        audio,
        sample_rate,
    )

    assert np.all(
        np.isfinite(features)
    )