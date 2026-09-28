from contextlib import contextmanager
from pathlib import Path
import os
import tempfile

import librosa
import numpy as np
from pydub import AudioSegment

from src.config import (
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    TARGET_SAMPLES,
    TRIM_TOP_DB,
)


class AudioProcessingError(Exception):
    """Raised when an audio file cannot be processed."""


def validate_audio_upload(
    filename: str,
    file_size: int,
) -> None:
    if not filename:
        raise AudioProcessingError("The uploaded file does not have a filename.")

    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise AudioProcessingError(
            f"Unsupported audio format: {extension}. "
            f"Supported formats: {supported}"
        )

    if file_size <= 0:
        raise AudioProcessingError("The uploaded audio file is empty.")

    max_size_bytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if file_size > max_size_bytes:
        raise AudioProcessingError(
            f"Audio file is too large. "
            f"Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )


@contextmanager
def temporary_audio_file(
    audio_bytes: bytes,
    suffix: str,
):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        yield temp_path

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def preprocess_audio(
    path: str,
) -> tuple[np.ndarray, int]:
    """
    Decode and preprocess audio for the existing emotion model.

    The current production model expects audio padded/truncated to
    TARGET_SAMPLES samples.

    This intentionally preserves the original application's use of
    raw PCM samples instead of silently changing the model input scale.
    """

    try:
        audio = AudioSegment.from_file(path)

    except Exception as exc:
        raise AudioProcessingError(
            "The audio file could not be decoded."
        ) from exc

    # Training datasets are speech/mono.
    # Convert stereo uploads to mono for predictable inference.
    if audio.channels > 1:
        audio = audio.set_channels(1)

    sample_rate = audio.frame_rate

    samples = np.asarray(
        audio.get_array_of_samples(),
        dtype=np.float32,
    )

    if samples.size == 0:
        raise AudioProcessingError(
            "No audio samples were found in the uploaded file."
        )

    trimmed, _ = librosa.effects.trim(
        samples,
        top_db=TRIM_TOP_DB,
    )

    if trimmed.size == 0:
        raise AudioProcessingError(
            "The uploaded audio does not contain enough audible signal."
        )

    if len(trimmed) < TARGET_SAMPLES:
        processed = np.pad(
            trimmed,
            (0, TARGET_SAMPLES - len(trimmed)),
            mode="constant",
        )

    else:
        processed = trimmed[:TARGET_SAMPLES]

    return processed.astype(np.float32), sample_rate