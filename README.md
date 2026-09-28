# Voice Emotion Recognition

A speech emotion recognition application that classifies emotional
characteristics from voice recordings using audio feature extraction
and a TensorFlow neural network.

The application analyzes a speech recording and predicts one of six
emotion classes:

- Neutral
- Happy
- Sad
- Angry
- Fear
- Disgust

## Demo

Live demo: Coming soon

## Overview

This project explores speech emotion recognition using acoustic
features extracted from raw audio.

The inference pipeline processes an uploaded recording, extracts
time-domain and spectral features, and passes the resulting feature
sequence to a trained TensorFlow model.

The application is currently deployed using Streamlit, while the
machine-learning inference code is intentionally framework-independent
so it can later be exposed through an API.

## Architecture

```text
Audio Upload
     |
     v
Audio Preprocessing
     |
     |-- silence trimming
     |-- mono conversion
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
TensorFlow Model
     |
     v
Emotion Prediction
     |
     v
Confidence Scores