from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.audio import preprocess_audio
from src.config import EMOTION_LABELS, MODEL_PATH
from src.features import extract_features
from src.schemas import PredictionResult


class PredictionError(Exception):
    """Raised when model inference fails."""


@lru_cache(maxsize=1)
def load_model(
    model_path: str = str(MODEL_PATH),
) -> tf.keras.Model:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {path}"
        )

    return tf.keras.models.load_model(
        path,
        compile=False,
    )


def _to_probabilities(
    output: np.ndarray,
) -> np.ndarray:
    """
    Preserve existing softmax outputs.

    If the model returns logits instead, convert them to probabilities.
    """

    output = np.asarray(
        output,
        dtype=np.float32,
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

    return exp_values / np.sum(exp_values)


def predict_audio(
    path: str,
) -> PredictionResult:
    try:
        audio, sample_rate = preprocess_audio(path)

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

    except Exception as exc:
        raise PredictionError(
            "Emotion prediction failed."
        ) from exc

    output = np.asarray(prediction)[0]

    if len(output) != len(EMOTION_LABELS):
        raise PredictionError(
            "The model output size does not match "
            "the configured emotion labels."
        )

    probabilities = _to_probabilities(output)

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
        emotion=EMOTION_LABELS[predicted_index],
        confidence=float(
            probabilities[predicted_index]
        ),
        scores=scores,
    )