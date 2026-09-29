import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.audio import (
    AudioProcessingError,
    preprocess_audio,
)
from src.config import (
    EMOTION_LABELS,
    EXPECTED_FRAME_COUNT,
    FEATURE_COUNT,
    MODEL_PATH,
)
from src.features import extract_features
from src.schemas import PredictionResult

logger = logging.getLogger(__name__)


class PredictionError(Exception):
    """Raised when model inference fails."""


def _validate_model_contract(
    model: tf.keras.Model,
) -> None:
    expected_input_shape = (
        None,
        EXPECTED_FRAME_COUNT,
        FEATURE_COUNT,
    )

    actual_input_shape = tuple(
        model.input_shape
    )

    if (
        actual_input_shape
        != expected_input_shape
    ):
        raise ValueError(
            "Unexpected model input shape: "
            f"{actual_input_shape}. "
            f"Expected {expected_input_shape}."
        )

    expected_output_size = len(
        EMOTION_LABELS
    )

    actual_output_size = int(
        model.output_shape[-1]
    )

    if (
        actual_output_size
        != expected_output_size
    ):
        raise ValueError(
            "Unexpected model output size: "
            f"{actual_output_size}. "
            f"Expected {expected_output_size}."
        )


@lru_cache(maxsize=1)
def load_model(
    model_path: str = str(MODEL_PATH),
) -> tf.keras.Model:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {path}"
        )

    model = tf.keras.models.load_model(
        path,
        compile=False,
    )

    _validate_model_contract(model)

    return model


def _to_probabilities(
    output: np.ndarray,
) -> np.ndarray:
    output = np.asarray(
        output,
        dtype=np.float32,
    )

    if not np.all(
        np.isfinite(output)
    ):
        raise ValueError(
            "Model output contains invalid numeric values."
        )

    if (
        np.all(output >= 0)
        and np.isclose(
            np.sum(output),
            1.0,
            atol=1e-3,
        )
    ):
        return output

    shifted = output - np.max(output)

    exp_values = np.exp(shifted)

    return (
        exp_values
        / np.sum(exp_values)
    )


def predict_audio(
    path: str,
) -> PredictionResult:
    try:
        audio, sample_rate = (
            preprocess_audio(path)
        )

        features = extract_features(
            audio,
            sample_rate,
        )

        model = load_model()

        prediction = model.predict(
            features,
            batch_size=1,
            verbose=0,
        )

    except AudioProcessingError:
        raise

    except Exception as exc:
        logger.exception(
            "Emotion prediction failed."
        )

        raise PredictionError(
            "Emotion prediction failed. "
            "Please try another recording."
        ) from exc

    output = np.asarray(
        prediction,
        dtype=np.float32,
    )[0]

    if (
        output.size
        != len(EMOTION_LABELS)
    ):
        raise PredictionError(
            "The model output does not match "
            "the configured emotion labels."
        )

    probabilities = _to_probabilities(
        output
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    scores = {
        label: float(probability)
        for label, probability in zip(
            EMOTION_LABELS,
            probabilities,
            strict=True,
        )
    }

    return PredictionResult(
        emotion=(
            EMOTION_LABELS[
                predicted_index
            ]
        ),
        confidence=float(
            probabilities[
                predicted_index
            ]
        ),
        scores=scores,
    )