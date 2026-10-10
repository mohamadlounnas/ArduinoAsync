# Lightweight Text-to-Speech (TTS) Architecture & Fine-Tuning for Algerian Darja

## Overview
This project provides a lightweight, CPU-efficient, end-to-end Text-to-Speech (TTS) training, fine-tuning, and inference framework specifically ported and tailored for **Algerian Darja (الدارجة الجزائرية)**.

Algerian Darja presents unique linguistic characteristics:
1. **Script dualism**: Written using Arabic script (e.g., "واش راك لاباس") or Arabizi/Latin script (e.g., "wach rak labas").
2. **Dialectal features**: Vowel shifts, French/Berber loanwords, and unique code-switching syntax.
3. **Resource constraints**: High-quality transcribed speech datasets are scarce, requiring lightweight models capable of fast fine-tuning on CPU/GPU environments.

---

## Model Architecture Selection

We adopt a modern **Lightweight Non-Autoregressive FastSpeech2 + Mel-Spectrogram Decoder + Griffin-Lim / Lightweight Vocoder** architecture:

1. **Phoneme / Character Encoder**:
   - Embedding Layer (vocab size ~128 covering Arabic characters, diacritics, Latin Arabizi numerals, and special tokens).
   - Lightweight 1D Convolutional Neural Network layers + LayerNorm + ReLU activation.
   - Positional Encodings to preserve sequence temporal structure.

2. **Variance Adaptor & Duration Predictor**:
   - Predicts duration per character/token to map text length $N$ to audio frame length $M$.
   - Dynamic Length Regulator expands text embeddings to match frame lengths without complex dynamic time warping.

3. **Mel-Spectrogram Decoder**:
   - Stacked lightweight 1D-CNN layers predicting 80-band log-mel spectrograms.
   - Mean Squared Error (MSE) loss on target vs predicted mel-spectrograms.

4. **Audio Synthesis (Vocoder)**:
   - Fast, CPU-friendly Griffin-Lim phase reconstruction / lightweight linear spectrogram inverter for real-time synthesis (Real-Time Factor < 0.1 on CPU).

---

## Text Normalization & Preprocessing for Algerian Darja

The preprocessing pipeline (`dataset_prep.py`) handles:
- **Diacritization & Unicode Normalization**: Strip or standardize Arabic tashkeel, normalize Alef variants (`أ, إ, آ` -> `ا`), normalize Teh Marbuta (`ة` -> `ه`).
- **Arabizi Mapping**: Convert Arabizi numbers used as letters (`3` -> `ع`, `7` -> `ح`, `9` -> `ق`, `5` -> `خ`, `2` -> `ء`) to standard phonemes/Arabic representations.
- **Vocabulary Mapping**: Character-to-Index encoding for seamless tokenization.

---

## Fine-Tuning & Iterative Loop

1. **Dataset**: Audio files sampled at 22050 Hz with aligned transcriptions in `dataset/metadata.csv`.
2. **Fine-Tuning Loop**:
   - Model parameters optimized with Adam optimizer ($\text{lr} = 10^{-3}$).
   - Loss function combines Mel-Spectrogram Reconstruction Loss (MSE) and Duration Predictor Loss.
3. **Evaluation**:
   - Real-Time Factor (RTF = synthesis time / audio duration).
   - Spectral distortion & Mel MSE loss.
   - Audio waveform verification (`.wav` generation).
