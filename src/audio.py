def preprocess_audio(
    path: str,
) -> tuple[np.ndarray, int]:
    try:
        _, sample_rate = librosa.load(
            path,
            sr=None,
        )

        raw_audio = AudioSegment.from_file(
            path
        )

    except Exception as exc:
        raise AudioProcessingError(
            "The audio file could not be decoded."
        ) from exc

    samples = np.asarray(
        raw_audio.get_array_of_samples(),
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

    if trimmed.size < TARGET_SAMPLES:
        processed = np.pad(
            trimmed,
            (
                0,
                TARGET_SAMPLES - trimmed.size,
            ),
            mode="constant",
        )
    else:
        processed = trimmed[:TARGET_SAMPLES]

    return (
        processed.astype(np.float32),
        sample_rate,
    )