from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT_DIR / "model" / "emotion_model.keras"

# Historical preprocessing contract used by the trained checkpoint.
#
# The reference training notebook calculated MFCC features using
# librosa's default 22,050 Hz sample rate while the PCM samples were
# obtained through pydub. Keep this value stable unless the model is
# retrained and revalidated.
FEATURE_SAMPLE_RATE = 22_050

# Audio preprocessing
TARGET_SAMPLES = 180_000
TRIM_TOP_DB = 25

# Feature extraction
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13

FEATURE_COUNT = N_MFCC + 2
EXPECTED_FRAME_COUNT = 1 + TARGET_SAMPLES // HOP_LENGTH

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
# This order must match the output neurons of the trained model.
EMOTION_LABELS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
)