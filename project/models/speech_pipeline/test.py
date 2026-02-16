# test.py
# Speech Emotion Recognition Testing Pipeline
# This script loads the trained model and preprocessed test data, performs evaluation, and generates reports/plots

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Scikit-learn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

print("✓ Libraries imported for testing!")

# Paths (must match train.py)
PREPROCESSED_DIR = "/content/preprocessed_data"
MODEL_DIR = "/content/models/speech_pipeline"

# Load preprocessed test data
X_test = np.load(f"{PREPROCESSED_DIR}/X_test.npy")
y_test = np.load(f"{PREPROCESSED_DIR}/y_test.npy")

with open(f"{PREPROCESSED_DIR}/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

print("Test data loaded!")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"Number of classes: {len(label_encoder.classes_)}")
print(f"Class names: {label_encoder.classes_}")

# Add channel dimension
X_test = X_test[..., np.newaxis]

# One-hot encode (for evaluation)
num_classes = len(label_encoder.classes_)
y_test_cat = to_categorical(y_test, num_classes)

# Load the trained model
model_path = f"{MODEL_DIR}/final_model.h5"
model = load_model(model_path)

print(f"✓ Model loaded from: {model_path}")
model.summary()

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Predictions
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

print(f"✓ Predictions generated for {len(y_pred)} samples")

# Classification Report
print("="*70)
print("CLASSIFICATION REPORT")
print("="*70)

report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    digits=4
)

print(report)

with open(f"{MODEL_DIR}/classification_report.txt", 'w') as f:
    f.write(report)

print(f"✓ Report saved to: {MODEL_DIR}/classification_report.txt")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot CM
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar_kws={'label': 'Count'}
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# Normalized CM
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt='.2%',
    cmap='Blues',
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
    cbar_kws={'label': 'Percentage'}
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix - Test Set')
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/confusion_matrix_normalized.png", dpi=150, bbox_inches='tight')
plt.show()

# Per-class accuracy
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

results_df = pd.DataFrame({
    'Emotion': label_encoder.classes_,
    'Accuracy': per_class_accuracy,
    'Correct': cm.diagonal(),
    'Total': cm.sum(axis=1)
}).sort_values('Accuracy', ascending=False)

print("="*70)
print("PER-CLASS ACCURACY")
print("="*70)
print(results_df.to_string(index=False))

# Plot per-class accuracy
plt.figure(figsize=(12, 6))
bars = plt.bar(results_df['Emotion'], results_df['Accuracy'],
               color='steelblue', edgecolor='black', alpha=0.7)

for i, (bar, acc) in enumerate(zip(bars, results_df['Accuracy'])):
    if acc >= 0.8:
        bar.set_color('green')
    elif acc >= 0.6:
        bar.set_color('orange')
    else:
        bar.set_color('red')
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
             f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')

plt.xlabel('Emotion')
plt.ylabel('Accuracy')
plt.title('Per-Class Accuracy on Test Set')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.axhline(y=test_accuracy, color='black', linestyle='--', linewidth=2, label=f'Overall: {test_accuracy:.2%}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/per_class_accuracy.png", dpi=150, bbox_inches='tight')
plt.show()

print(f"\nEasiest: {results_df.iloc[0]['Emotion']} ({results_df.iloc[0]['Accuracy']:.2%})")
print(f"Hardest: {results_df.iloc[-1]['Emotion']} ({results_df.iloc[-1]['Accuracy']:.2%})")

# Error Analysis
misclassified_indices = np.where(y_pred != y_test)[0]

print("="*70)
print("ERROR ANALYSIS")
print("="*70)
print(f"Total misclassified: {len(misclassified_indices)} / {len(y_test)} ({len(misclassified_indices)/len(y_test)*100:.2f}%)")

print("\nSample Misclassifications (first 10):")
for i, idx in enumerate(misclassified_indices[:10]):
    true_label = label_encoder.classes_[y_test[idx]]
    pred_label = label_encoder.classes_[y_pred[idx]]
    confidence = y_pred_probs[idx][y_pred[idx]] * 100
    print(f"{i+1:2d}. True: {true_label:20s} | Pred: {pred_label:20s} | Conf: {confidence:.1f}%")

# Common confusions
confusion_pairs = []
for i in range(len(label_encoder.classes_)):
    for j in range(len(label_encoder.classes_)):
        if i != j and cm[i, j] > 0:
            confusion_pairs.append((label_encoder.classes_[i], label_encoder.classes_[j], cm[i, j]))

confusion_pairs.sort(key=lambda x: x[2], reverse=True)

print("\nTop 5 Confusion Patterns:")
for i, (true_l, pred_l, count) in enumerate(confusion_pairs[:5]):
    print(f"{i+1}. {true_l:20s} → {pred_l:20s}: {count} times")

# Save predictions
np.save(f"{MODEL_DIR}/test_predictions.npy", y_pred)
np.save(f"{MODEL_DIR}/test_predictions_probs.npy", y_pred_probs)
print(f"✓ Predictions saved to {MODEL_DIR}")

print("="*70)
print("TESTING PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Results saved in: {MODEL_DIR}")