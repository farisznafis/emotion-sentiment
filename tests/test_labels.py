from src.config import EMOTION_LABELS


def test_model_has_six_emotion_labels():
    assert len(EMOTION_LABELS) == 6


def test_emotion_label_order_is_preserved():
    assert EMOTION_LABELS == (
        "neutral",
        "happy",
        "sad",
        "angry",
        "fear",
        "disgust",
    )


def test_emotion_labels_are_unique():
    assert len(
        set(EMOTION_LABELS)
    ) == len(
        EMOTION_LABELS
    )