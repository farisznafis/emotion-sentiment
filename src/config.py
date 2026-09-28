from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "model" / "emotion_model.keras"

# Audio preprocessing
TARGET_SAMPLES = 180_000
TRIM_TOP_DB = 25

# Feature extraction
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13

# Upload
MAX_UPLOAD_SIZE_MB = 15

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".ogg",
    ".aac",
    ".flac",
    ".amr",
    ".m4a",
}

# IMPORTANT:
# This order must match the output neurons of the existing trained model.
EMOTION_LABELS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
)