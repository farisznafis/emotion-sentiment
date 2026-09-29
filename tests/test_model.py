import numpy as np

from src.config import (
    EMOTION_LABELS,
    EXPECTED_FRAME_COUNT,
    FEATURE_COUNT,
)
from src.predictor import load_model


def test_model_has_expected_contract():
    model = load_model()

    assert tuple(
        model.input_shape
    ) == (
        None,
        EXPECTED_FRAME_COUNT,
        FEATURE_COUNT,
    )

    assert tuple(
        model.output_shape
    ) == (
        None,
        len(EMOTION_LABELS),
    )


def test_model_accepts_expected_feature_shape():
    model = load_model()

    features = np.zeros(
        (
            1,
            EXPECTED_FRAME_COUNT,
            FEATURE_COUNT,
        ),
        dtype=np.float32,
    )

    prediction = model.predict(
        features,
        verbose=0,
    )

    assert prediction.shape == (
        1,
        len(EMOTION_LABELS),
    )

    assert np.all(
        np.isfinite(prediction)
    )

    assert np.isclose(
        np.sum(prediction[0]),
        1.0,
        atol=1e-3,
    )