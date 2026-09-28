import librosa
import numpy as np

from src.config import (
    FRAME_LENGTH,
    HOP_LENGTH,
    N_MFCC,
)


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    Extract the features expected by the trained model.

    Features per frame:
    - Zero Crossing Rate: 1
    - RMS Energy: 1
    - MFCC: 13

    Total:
    15 features per frame
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

    # Defensive alignment in case individual feature extractors
    # return slightly different frame counts.
    min_frames = min(
        zcr.shape[1],
        rms.shape[1],
        mfcc.shape[1],
    )

    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]
    mfcc = mfcc[:, :min_frames]

    features = np.concatenate(
        (zcr, rms, mfcc),
        axis=0,
    )

    # (features, frames) -> (frames, features)
    features = features.T

    # Add batch dimension:
    # (frames, features) -> (1, frames, features)
    features = np.expand_dims(
        features,
        axis=0,
    )

    if not np.all(np.isfinite(features)):
        raise ValueError(
            "Feature extraction produced invalid numeric values."
        )

    return features.astype(np.float32)