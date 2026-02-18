# Multimodal Emotion Recognition on TESS Dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-green.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A comprehensive multimodal emotion recognition system using speech, text, and late fusion on the Toronto Emotional Speech Set (TESS).**

This project implements three pipelines as per **Assignment 2: Multimodal Emotion Recognition**:
- **Speech-only**: CNN-BiLSTM and BiLSTM with attention for acoustic emotion cues.
- **Text-only**: TF-IDF + linear projection for lexical analysis.
- **Multimodal (Late Fusion)**: Combines speech and text embeddings for unified classification.

Built to evaluate the dominance of acoustic features in acted emotional speech, with near-ceiling performance on TESS.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Architectures](#architectures)
- [Results](#results)
- [Analysis](#analysis)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Report](#report)
- [Future Work](#future-work)
- [References](#references)
- [Contributing](#contributing)

---

## 🎯 Overview

Speech Emotion Recognition (SER) is crucial for affective computing applications like mental health monitoring and human-AI interaction. This project systematically compares unimodal (speech/text) and multimodal approaches on the **TESS dataset**, which features highly controlled, lexically neutral utterances.

**Key Research Questions Addressed**:
1. How effective are acoustic features alone?
2. Does text add value in fusion?
3. What are the class-wise challenges?
4. How do embeddings cluster?

**Main Findings** (from experiments):
- Speech dominates (99.46% accuracy).
- Text is weak (55.86%).
- Fusion slightly degrades performance (96.43%) due to noisy text signals.

---

## 📊 Dataset

**Toronto Emotional Speech Set (TESS)**:
- ~2,800 utterances from 2 female actors (OAF, YAF).
- 7 balanced emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `pleasant surprise`, `sad`.
- Fixed carrier phrases: *"Say the word [WORD]"*.
- Splits: Stratified 80/10/10 (train/val/test) + speaker-independent (OAF train, YAF test).

**Preprocessing**:
- **Speech**: Resample (16-22kHz), silence trim, normalize, extract Mel-spectrograms/MFCCs + prosodic features.
- **Text**: Extract carrier phrases from filenames, TF-IDF vectorization (train-only).

**Download**: [Kaggle TESS](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) (or provided link in assignment).

---

## 📁 Project Structure

```bash
project/
├── models/
│   ├── speech_pipeline/
│   │   ├── train.py          # CNN-BiLSTM & BiLSTM-Attention
│   │   └── test.py
│   ├── text_pipeline/
│   │   ├── train.py          # TF-IDF + Linear
│   │   └── test.py
│   └── fusion_pipeline/
│       ├── train.py          # Late fusion (1024-dim concat)
│       └── test.py
├── results/
│   ├── accuracy_tables/      # All model results (CSV/PNG)
│   └── plots/                # t-SNE, confusion matrices, bar charts
├── README.md
├── requirements.txt
├── report.pdf                # Full analysis (Architecture, Experiments, Error cases)
└── data/                     # (Gitignored) TESS audio + transcripts
```

---

## 🏗️ Architectures

### 1. Speech-Only
- **CNN-BiLSTM** (TensorFlow/Keras): 3 conv blocks → BiLSTM (128→64) → FC classifier.
- **BiLSTM w/ Attention** (PyTorch): 2-layer BiLSTM + multi-pooling (mean/max/last) + MLP.
- **Why?** Captures temporal prosody and spectral patterns in emotional speech.
  
[![Sound Model](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DTSwQfZFra2471RYWXRjtdemgXHSMPA5?usp=sharing)

### 2. Text-Only
- **TF-IDF** (scikit-learn) → Linear projection (512-dim).
- **Why?** Simple baseline for lexical cues (limited in TESS due to uniform phrases).
  
[![Text Model](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KVnR4qNR0R91BhSwKB3MH5FyiyWTX0BG?usp=sharing)

### 3. Multimodal Late Fusion
- Speech: 512-dim embedding.
- Text: 512-dim projection.
- **Fusion**: Concat (1024-dim) → FC (1024→512→7) w/ dropout.
- **Why?** Combines high-level features; tests complementarity.
  
[![Fusion Model](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1z-RO0-bGepgDyvUsIhD6v5fyHMmVoRi6?usp=sharing)

**Training**:
- Optimizer: Adam/AdamW (LR: 1e-4).
- Loss: Cross-entropy.
- Early stopping + LR scheduling.
- Metrics: Accuracy, Weighted F1.

---

## 📈 Results

### Overall Performance

| Model Configuration | Test Accuracy | Weighted F1 | Remarks                  |
|---------------------|---------------|-------------|--------------------------|
| **Speech-only**    | **99.46%**   | ~99.4%     | Near-perfect             |
| **Text-only**      | 55.86%       | ~55.0%     | Above random (7 classes) |
| **Fusion**         | 96.43%       | ~96.4%     | Slight degradation       |

**Visuals**:
- [t-SNE Embeddings](results/plots/t-sne_speech_text_fusion.png) (Speech: Tight clusters; Text: Overlap; Fusion: Speech-dominated).
- [Confusion Matrices](results/plots/confusion_matrices.png).

### Class-wise Insights
- **Easiest**: Neutral, Angry, Sad (distinct prosody: flat/high-energy/low-pitch).
- **Hardest**: Disgust, Fear, Pleasant Surprise (acoustic overlap: high pitch, tense voice).
- **Common Errors**: Fear→Angry, Disgust→Sad, Pleasant Surprise→Happy.

---

## 🔍 Analysis

### Error Analysis (3-5 Failure Cases)
1. **Fear misclassified as Angry**: Shared high arousal (abrupt energy spikes).
2. **Disgust as Sad**: Subtle low-energy overlap in acted speech.
3. **Pleasant Surprise → Happy**: Semantic/acoustic similarity (rising pitch).
4. **Fusion Degradation**: Text noise blurs speech clusters (e.g., fear-disgust).

### When Does Fusion Help?
- **Rarely on TESS**: Lexically neutral data makes text uninformative.
- **Potential**: In naturalistic datasets with rich transcripts.

### Embedding Separability (t-SNE)
- **Temporal (Speech)**: Excellent clusters.
- **Contextual (Text)**: Poor structure.
- **Fusion**: Mirrors speech, minor blurring in confusable pairs.

**Full Report**: See `report.pdf` for architecture decisions, experiments, and visualizations.

---

## 🚀 Setup & Installation

1. **Clone Repo**:
   ```bash
   git clone https://github.com/yourusername/multimodal-emotion-tess.git
   cd multimodal-emotion-tess
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Dataset**:
   - Download TESS to `data/tess/`.
   - Run preprocessing scripts (in pipelines).

**Requirements** (`requirements.txt`):
- `torch torchvision torchaudio`
- `tensorflow keras`
- `scikit-learn`
- `librosa pandas numpy`
- `matplotlib seaborn plotly`
- `scikit-learn`

---

## ▶️ Usage

### Train & Test Pipelines

```bash
# Speech-only
python models/speech_pipeline/train.py --model cnn_bilstm
python models/speech_pipeline/test.py --checkpoint best_model.pt

# Text-only
python models/text_pipeline/train.py
python models/text_pipeline/test.py

# Fusion
python models/fusion_pipeline/train.py
python models/fusion_pipeline/test.py
```

**Outputs**:
- Models saved to `checkpoints/`.
- Results/plots to `results/`.

**Speaker-Independent**:
- Use `--speaker_independent` flag (OAF train, YAF test).

---

## 📝 Report

**A. Architecture Decisions**:
- Detailed rationale in `report.pdf`.

**B. Experiments**:
- 3 variants compared (tables/plots).

**C. Analysis**:
- Easiest/hardest emotions.
- Fusion insights.
- 5 failure cases w/ audio samples.
- t-SNE visuals.

---

## 🔮 Future Work

- **Cross-modal Attention**: Better fusion (e.g., transformers).
- **Naturalistic Datasets**: CREMA-D, IEMOCAP (richer text).
- **Real-time Inference**: Edge deployment.
- **Multilingual**: Extend to non-English carriers.

---

## 📚 References

1. **TESS Dataset**: [Kaggle](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess).
2. **Paper Inspiration**: "Multimodal Emotion Recognition on the TESS Dataset Using Speech, Text, and Late Fusion" (attached PDF).
3. **Libraries**: Librosa, PyTorch, TensorFlow.

---

## 🤝 Contributing

Contributions welcome! Open an issue or PR for:
- New fusion strategies.
- Dataset expansions.
- UI demo (Streamlit/Gradio).


---

*Built for Assignment 2 | Last Updated: Feb 2026*
