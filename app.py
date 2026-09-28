from pathlib import Path

import pandas as pd
import streamlit as st

from src.audio import (
    AudioProcessingError,
    temporary_audio_file,
    validate_audio_upload,
)
from src.config import MAX_UPLOAD_SIZE_MB
from src.predictor import (
    PredictionError,
    predict_audio,
)


st.set_page_config(
    page_title="Voice Emotion Recognition",
    page_icon="🎙️",
    layout="centered",
)


def render_header() -> None:
    st.title("Voice Emotion Recognition")

    st.write(
        """
        Analyze emotional characteristics from a short speech recording
        using a TensorFlow-based speech emotion recognition model.
        """
    )

    st.caption(
        "Supports WAV, MP3, M4A, FLAC, OGG, AAC and AMR."
    )


def render_prediction(result) -> None:
    st.success("Analysis complete")

    emotion_column, confidence_column = st.columns(2)

    with emotion_column:
        st.metric(
            label="Detected emotion",
            value=result.emotion.title(),
        )

    with confidence_column:
        st.metric(
            label="Model confidence",
            value=f"{result.confidence:.1%}",
        )

    st.progress(
        min(
            max(result.confidence, 0.0),
            1.0,
        )
    )

    st.subheader("Confidence by emotion")

    dataframe = pd.DataFrame(
        {
            "Emotion": [
                emotion.title()
                for emotion in result.scores
            ],
            "Confidence": [
                score
                for score in result.scores.values()
            ],
        }
    )

    dataframe["Confidence (%)"] = (
        dataframe["Confidence"] * 100
    )

    dataframe = dataframe.sort_values(
        "Confidence",
        ascending=False,
    )

    st.bar_chart(
        dataframe.set_index("Emotion")[
            "Confidence"
        ]
    )

    with st.expander("View detailed scores"):
        st.dataframe(
            dataframe[
                [
                    "Emotion",
                    "Confidence (%)",
                ]
            ],
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    render_header()

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload a speech recording",
        type=[
            "wav",
            "mp3",
            "ogg",
            "aac",
            "flac",
            "amr",
            "m4a",
        ],
        help=(
            "For better results, use clear speech "
            "with minimal background noise."
        ),
    )

    if uploaded_file is None:
        st.info(
            "Upload an audio recording to start the analysis."
        )

        return

    audio_bytes = uploaded_file.getvalue()

    try:
        validate_audio_upload(
            filename=uploaded_file.name,
            file_size=len(audio_bytes),
        )

    except AudioProcessingError as exc:
        st.error(str(exc))

        return

    st.audio(audio_bytes)

    st.caption(
        f"{uploaded_file.name} · "
        f"{len(audio_bytes) / 1024 / 1024:.2f} MB"
    )

    analyze_button = st.button(
        "Analyze emotion",
        type="primary",
        use_container_width=True,
    )

    if not analyze_button:
        return

    suffix = (
        Path(uploaded_file.name)
        .suffix
        .lower()
    )

    try:
        with st.spinner(
            "Analyzing your recording..."
        ):
            with temporary_audio_file(
                audio_bytes,
                suffix,
            ) as temp_path:
                result = predict_audio(
                    temp_path
                )

    except (
        AudioProcessingError,
        PredictionError,
    ) as exc:
        st.error(str(exc))

        return

    except Exception:
        st.error(
            "An unexpected error occurred while "
            "processing the audio."
        )

        return

    render_prediction(result)

    st.divider()

    st.caption(
        "This project is an experimental machine-learning demo. "
        "Emotion predictions are model estimates and should not be "
        "used as psychological, medical, or behavioral assessments."
    )


if __name__ == "__main__":
    main()