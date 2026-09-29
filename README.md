# Voice Emotion Recognition

An end-to-end Speech Emotion Recognition (SER) project that converts
speech recordings into acoustic feature sequences and classifies them
with a TensorFlow/Keras LSTM model.

The application predicts one of six classes:

- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust

The current repository separates the historical research workflow from
the production inference path so the model can be inspected, tested,
and deployed without depending on a notebook runtime.

## Overview

The project started as an audio-classification experiment built from
multiple emotional-speech datasets and later evolved into a deployable
Streamlit application.

The full pipeline is:

```text
Audio file
    |
    v
Upload validation
    |
    v
Audio decoding
    |
    v
Silence trimming
    |
    v
Pad / truncate to 180,000 PCM samples
    |
    v
Feature extraction
    |
    |-- Zero Crossing Rate: 1
    |-- RMS Energy: 1
    |-- MFCC: 13
    |
    v
Feature tensor
(batch, 352, 15)
    |
    v
LSTM 64
    |
    v
LSTM 64
    |
    v
Dense softmax 6
    |
    v
Emotion class + confidence scores
```

## Data

The historical training notebook combines four emotional-speech
datasets into one six-class label space:

- RAVDESS
- CREMA-D
- TESS
- SAVEE

The notebook parses each dataset's original label convention into:

```text
neutral
happy
sad
angry
fear
disgust
```

It also records speaker gender and inspects class/gender distribution.
The recorded experiment then filters the working dataframe to female
speech before feature extraction.

This filtering decision is important when interpreting model
generalization: the experiment should not be treated as a universal
speech-emotion model across all speaker populations.

## Methodology

### 1. Audio preprocessing

The historical inference behavior is intentionally preserved for
compatibility with the deployed checkpoint.

For each recording:

1. Decode the file.
2. Read the original sample rate with `librosa.load(..., sr=None)`.
3. Obtain raw PCM samples through Pydub.
4. Trim low-energy leading/trailing regions with `top_db=25`.
5. Pad or truncate the resulting array to exactly 180,000 samples.
6. Keep the original sample rate when calculating MFCC features.

The production code deliberately does not introduce a new resampling
policy because changing MFCC sample-rate handling was found to alter
model confidence scores even when the final tensor shape remained
valid.

### 2. Acoustic feature extraction

Each frame is represented by 15 acoustic values:

| Feature | Dimensions |
| --- | ---: |
| Zero Crossing Rate | 1 |
| RMS Energy | 1 |
| MFCC | 13 |
| **Total** | **15** |

Feature parameters:

| Parameter | Value |
| --- | ---: |
| Target PCM length | 180,000 samples |
| Frame length | 2,048 |
| Hop length | 512 |
| MFCC coefficients | 13 |
| Frames | 352 |
| Model input | `(batch, 352, 15)` |

The features capture complementary aspects of speech:

- **Zero Crossing Rate** describes rapid sign changes in the waveform.
- **RMS Energy** summarizes signal energy over time.
- **MFCCs** approximate the spectral envelope used extensively in speech
  processing.

### 3. Data split

The reference notebook performs:

```python
train_test_split(
    X,
    y,
    test_size=0.12,
    random_state=1,
)
```

The reserved 12% is then split again with a 70/30 validation/test
division.

That produces approximately:

- 88% training
- 8.4% validation
- 3.6% test

The historical validation classification report contains 263 samples.

## Model Architecture

The reference network is a compact stacked LSTM classifier:

```text
Input
(None, 352, 15)
    |
    v
LSTM(64, return_sequences=True)
    |
    v
LSTM(64)
    |
    v
Dense(6, softmax)
```

Model summary:

| Layer | Output shape | Parameters |
| --- | --- | ---: |
| LSTM | `(None, 352, 64)` | 20,480 |
| LSTM | `(None, 64)` | 33,024 |
| Dense | `(None, 6)` | 390 |
| **Total** |  | **53,894** |

The reference training configuration uses:

- categorical cross-entropy
- RMSProp
- categorical accuracy
- batch size 6
- up to 400 epochs

## Historical Evaluation

The reference notebook records a validation classification report with:

- validation samples: **263**
- accuracy: **0.94**
- macro F1: approximately **0.93**
- weighted F1: approximately **0.94**

These numbers are useful as historical experimental evidence, but they
should **not** be interpreted as a reproduced production benchmark for
the currently deployed checkpoint.

### Why the metric is presented with caution

During the portfolio remaster, several provenance and evaluation issues
were identified.

#### 1. Display-label mismatch

The training class mapping is:

```text
0 neutral
1 happy
2 sad
3 angry
4 fear
5 disgust
```

However, the historical confusion-matrix / classification-report labels
were written as:

```text
neutral
calm
sad
happy
fear
disgust
```

Therefore the textual names shown for class indices 1 and 3 in that
historical report are incorrect.

The overall index-based accuracy remains interpretable, but per-class
conclusions should be regenerated with corrected labels.

#### 2. Callback metric mismatch

The model is compiled with:

```text
categorical_accuracy
```

while the recorded EarlyStopping and ModelCheckpoint callbacks monitor:

```text
val_accuracy
```

The notebook log explicitly reports that `val_accuracy` is unavailable
and that the callbacks cannot operate as configured.

Future training should monitor:

```text
val_categorical_accuracy
```

instead.

#### 3. Training notebook checkpoint and deployed checkpoint differ

The current application uses the model binary historically stored as:

```text
best_model_22112024_400.keras
```

and renamed in the cleaned repository to:

```text
model/emotion_model.keras
```

The reference notebook later trains/saves a different 03122024 model.

Because they are different checkpoints, evaluation results from the
reference notebook should not automatically be attributed to the
deployed model.

## Production Inference

The deployable application is intentionally separated from the notebook:

```text
app.py
src/
├── audio.py
├── config.py
├── features.py
├── predictor.py
└── schemas.py
```

Responsibilities:

- `audio.py`
  - upload validation
  - temporary-file lifecycle
  - decoding
  - silence trimming
  - fixed-length preprocessing

- `features.py`
  - ZCR
  - RMS
  - MFCC
  - shape validation

- `predictor.py`
  - Keras model loading
  - model input/output contract validation
  - inference
  - probability handling
  - prediction result construction

- `app.py`
  - Streamlit UI
  - playback
  - inference trigger
  - detected class
  - confidence visualization
  - user-facing errors

## Model Contract

The current runtime validates the assumptions around the stored model.

Expected input:

```text
(None, 352, 15)
```

Expected output:

```text
(None, 6)
```

Emotion order:

```text
neutral
happy
sad
angry
fear
disgust
```

Keeping this order stable is essential because each output neuron maps
directly to one configured emotion label.

## Testing and CI

The remastered repository adds automated checks around the parts that
are easiest to break during an ML refactor.

Current tests cover:

- upload validation
- fixed preprocessing length
- preservation of source sample rate
- finite acoustic features
- expected feature tensor shape
- label count and ordering
- Keras model loading
- model input/output contract
- synthetic inference output

GitHub Actions runs on Linux and performs:

```text
Ruff lint
    |
Ruff format check
    |
Pytest
    |
model load
    |
synthetic TensorFlow inference
```

This also gives the project an OS-independent validation path for
native Python dependencies: the same inference contract is checked in
Linux CI and in the Streamlit deployment environment as well as through
the source-level tests.

## Supported Audio Formats

The application accepts:

- WAV
- MP3
- M4A
- FLAC
- OGG
- AAC
- AMR

Compressed formats depend on FFmpeg at runtime.

## Tech Stack

### Machine learning

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Pydub
- SoundFile

### Application

- Streamlit

### Quality

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
│   ├── __init__.py
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

### 1. Clone

```bash
git clone https://github.com/farisznafis/emotion-sentiment.git
cd emotion-sentiment
git checkout main-v2
```

### 2. Create an environment

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

Development tools:

```bash
python -m pip install -r requirements-dev.txt
```

### 4. Run

```bash
python -m streamlit run app.py
```

## Development

Lint:

```bash
ruff check app.py src tests
```

Format:

```bash
ruff format app.py src tests
```

Test:

```bash
python -m pytest
```

## Limitations

This project is an experimental machine-learning system, not an
objective emotion detector.

Important limitations include:

- the reference experiment filters the working data to female speech;
- the source datasets use acted emotional speech and different recording
  conditions;
- language, speaker, microphone, noise, and recording format can affect
  predictions;
- softmax scores are shown as model confidence but have not been
  calibrated as probabilities of correctness;
- the historical notebook and deployed checkpoint do not currently form
  one fully reproduced evaluation artifact;
- the inference pipeline preserves legacy preprocessing for checkpoint
  compatibility rather than enforcing a newly standardized resampling
  policy.

The application should not be used for psychological, medical,
employment, behavioral, or other high-stakes assessments.

## What I Would Improve Next

A rigorous next iteration would:

1. define one explicit audio sample rate and resample all recordings;
2. retrain the model against that documented preprocessing contract;
3. correct the historical evaluation label names;
4. monitor `val_categorical_accuracy` consistently in callbacks;
5. bind evaluation to the exact checkpoint that is deployed;
6. report a reproducible held-out test confusion matrix;
7. add per-class precision, recall, and F1 with correct labels;
8. evaluate calibration instead of treating softmax directly as
   probability;
9. test generalization across speakers, genders, languages, microphones,
   and recording environments;
10. compare the stacked LSTM baseline with stronger audio encoders or
    modern pretrained speech representations.

## Training Reference

The original research workflow is preserved in:

```text
notebooks/training_reference.ipynb
```

It is retained intentionally as historical experiment documentation.
Production inference lives separately under `src/`.

That separation makes it possible to improve application engineering
without rewriting the historical experiment or silently changing the
numerical behavior expected by the deployed checkpoint.
