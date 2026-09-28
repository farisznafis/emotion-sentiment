# Voice Emotion Recognition

A speech emotion recognition application that estimates emotional
characteristics from voice recordings using acoustic feature extraction
and a TensorFlow LSTM model.

The application predicts one of six emotion classes:

- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust

## Live Demo

https://YOUR-APP.streamlit.app

## Overview

This project explores Speech Emotion Recognition (SER) using acoustic
features extracted from voice recordings.

An uploaded audio recording is preprocessed into a fixed-length signal.
Zero Crossing Rate, RMS Energy, and MFCC features are then extracted and
passed to a trained TensorFlow model.

The Streamlit application provides an interactive interface for running
inference and inspecting confidence scores for each emotion class.

## Architecture

```text
Audio Upload
     |
     v
Audio Validation
     |
     v
Audio Preprocessing
     |
     |-- decode audio
     |-- mono conversion
     |-- silence trimming
     |-- padding / truncation
     |
     v
Feature Extraction
     |
     |-- Zero Crossing Rate
     |-- RMS Energy
     |-- 13 MFCC coefficients
     |
     v
Feature Tensor
(1, 352, 15)
     |
     v
TensorFlow LSTM Model
     |
     v
6 Emotion Probabilities
     |
     v
Prediction + Confidence Scores
```

## Model Input

The deployed checkpoint expects a fixed feature tensor:

| Property | Value |
| --- | --- |
| Audio samples | 180,000 |
| Feature sample rate | 22,050 Hz |
| Sequence frames | 352 |
| Features per frame | 15 |
| ZCR | 1 |
| RMS | 1 |
| MFCC | 13 |
| Output classes | 6 |

The feature sample rate is intentionally kept compatible with the
historical preprocessing pipeline used by the reference training
notebook.

Changing this preprocessing contract should be accompanied by model
retraining or validation.

## Tech Stack

- Python 3.12
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Pydub
- SoundFile
- Streamlit
- Pytest
- Ruff
- GitHub Actions

## Project Structure

```text
emotion-sentiment/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .streamlit/
│   └── config.toml
├── model/
│   └── emotion_model.keras
├── notebooks/
│   └── training_reference.ipynb
├── src/
│   ├── audio.py
│   ├── config.py
│   ├── features.py
│   ├── predictor.py
│   └── schemas.py
├── tests/
│   ├── test_audio.py
│   ├── test_features.py
│   ├── test_labels.py
│   └── test_model.py
├── app.py
├── packages.txt
├── pyproject.toml
├── requirements-dev.txt
└── requirements.txt
```

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/farisznafis/emotion-sentiment.git
cd emotion-sentiment
git checkout main-v2
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development:

```bash
python -m pip install -r requirements-dev.txt
```

FFmpeg is also required for decoding compressed audio formats.

### 4. Run the application

```bash
python -m streamlit run app.py
```

## Development

Run tests:

```bash
python -m pytest
```

Run linting:

```bash
ruff check app.py src tests
```

Check formatting:

```bash
ruff format --check app.py src tests
```

Automatically format the source code:

```bash
ruff format app.py src tests
```

## Supported Audio Formats

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- AMR

## Limitations

Speech emotion recognition estimates acoustic patterns associated with
the model's training labels. Emotion is contextual and cannot be
reliably determined from vocal characteristics alone.

Predictions may vary depending on recording quality, background noise,
speaker characteristics, language, microphone, and other factors.

This application is an experimental machine-learning project and is not
intended for psychological, medical, employment, or behavioral
assessment.

## Training Reference

The original model-development workflow is preserved in:

```text
notebooks/training_reference.ipynb
```

The notebook is retained as a training reference, while production
inference code is maintained separately under `src/`.