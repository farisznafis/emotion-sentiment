import numpy as np
import pytest
import soundfile as sf

from src.audio import (
    AudioProcessingError,
    preprocess_audio,
    validate_audio_upload,
)
from src.config import (
    FEATURE_SAMPLE_RATE,
    MAX_UPLOAD_SIZE_MB,
    TARGET_SAMPLES,
)


def test_validate_audio_upload_accepts_wav():
    validate_audio_upload(
        filename="sample.wav",
        file_size=1024,
    )


def test_validate_audio_upload_rejects_unknown_format():
    with pytest.raises(
        AudioProcessingError
    ):
        validate_audio_upload(
            filename="sample.exe",
            file_size=1024,
        )


def test_validate_audio_upload_rejects_empty_file():
    with pytest.raises(
        AudioProcessingError
    ):
        validate_audio_upload(
            filename="sample.wav",
            file_size=0,
        )


def test_validate_audio_upload_rejects_large_file():
    too_large = (
        MAX_UPLOAD_SIZE_MB
        * 1024
        * 1024
        + 1
    )

    with pytest.raises(
        AudioProcessingError
    ):
        validate_audio_upload(
            filename="sample.wav",
            file_size=too_large,
        )


def test_preprocess_audio_returns_model_contract(
    tmp_path,
):
    source_sample_rate = 44_100

    time = (
        np.arange(
            source_sample_rate,
            dtype=np.float32,
        )
        / source_sample_rate
    )

    audio = (
        0.25
        * np.sin(
            2
            * np.pi
            * 440
            * time
        )
    ).astype(np.float32)

    audio_path = (
        tmp_path
        / "sample.wav"
    )

    sf.write(
        audio_path,
        audio,
        source_sample_rate,
    )

    processed, feature_sample_rate = (
        preprocess_audio(
            str(audio_path)
        )
    )

    assert processed.shape == (
        TARGET_SAMPLES,
    )

    assert (
        feature_sample_rate
        == source_sample_rate
    )

    assert np.all(
        np.isfinite(processed)
    )