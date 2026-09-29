import librosa
import numpy as np

from src.config import (
    EXPECTED_FRAME_COUNT,
    FEATURE_COUNT,
    FRAME_LENGTH,
    HOP_LENGTH,
    N_MFCC,
)


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Extract acoustic features expected by the trained model.

    Per frame:
    - Zero Crossing Rate: 1
    - RMS Energy: 1
    - MFCC: 13

    Final model input:
    (1, 352, 15)
    """

    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH,
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
    )

    min_frames = min(
        zcr.shape[1],
        rms.shape[1],
        mfcc.shape[1],
    )

    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]
    mfcc = mfcc[:, :min_frames]

    features = np.concatenate(
        (
            zcr,
            rms,
            mfcc,
        ),
        axis=0,
    )

    features = features.T

    features = np.expand_dims(
        features,
        axis=0,
    ).astype(np.float32)

    if not np.all(
        np.isfinite(features)
    ):
        raise ValueError(
            "Feature extraction produced invalid numeric values."
        )

    expected_shape = (
        1,
        EXPECTED_FRAME_COUNT,
        FEATURE_COUNT,
    )

    if features.shape != expected_shape:
        raise ValueError(
            "Unexpected feature shape: "
            f"{features.shape}. "
            f"Expected {expected_shape}."
        )

    return features