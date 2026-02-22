# train.py
# ================================================
# TESS Emotion Recognition - TRAIN SCRIPT
# BiLSTM with Attention + Rich Features
# Speaker-independent split (OAF train, YAF test)
# ================================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 8

FEATURE_KEY = 'rich_feature_sequence'   # Options: 'rich_feature_sequence' or 'mel_sequence'

EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTION_LABELS)

# ── Custom Dataset ────────────────────────────────────────────────────────
class TessSequenceDataset(Dataset):
    def __init__(self, dataframe, feature_key, label_encoder):
        self.features = [np.array(row[feature_key], dtype=np.float32) for _, row in dataframe.iterrows()]
        self.labels   = label_encoder.transform(dataframe['emotion'].values)
        self.features = np.stack(self.features)   # (N, seq_len, feat_dim)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )


# ── BiLSTM with Attention ─────────────────────────────────────────────────
class EmotionBiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=7, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Bidirectional → output dim per time step = hidden_size × 2
        lstm_out_dim = hidden_size * 2

        self.attention = nn.Linear(lstm_out_dim, 1)

        # Four pooling strategies → 4 × lstm_out_dim
        concat_dim = lstm_out_dim * 4

        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                      # (B, seq, hidden*2)

        # Attention
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)     # (B, seq, 1)
        attn_context = torch.sum(lstm_out * attn_w, dim=1)          # (B, hidden*2)

        # Pooling features
        mean_pool  = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        last_hidden = lstm_out[:, -1, :]

        combined = torch.cat([mean_pool, max_pool, last_hidden, attn_context], dim=1)
        return self.fc(combined)


# ── Train & Evaluate ──────────────────────────────────────────────────────
def train_and_evaluate():
    print(f"Using feature: {FEATURE_KEY}")
    print(f"Device: {DEVICE}\n")

    # Load data (relative paths - place pkl files in the same directory)
    train_df = pd.read_pickle("train_speaker_split.pkl")
    test_df  = pd.read_pickle("test_speaker_split.pkl")

    # Label encoder (fixed order)
    le = LabelEncoder()
    le.fit(EMOTION_LABELS)

    train_ds = TessSequenceDataset(train_df, FEATURE_KEY, le)
    test_ds  = TessSequenceDataset(test_df, FEATURE_KEY, le)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)}")
    print(f"Classes: {list(le.classes_)}\n")

    # Quick shape check
    sample_x, _ = train_ds[0]
    print(f"Input shape example: {sample_x.shape}  →  seq_len × features\n")

    # Model
    input_dim = train_ds[0][0].shape[-1]
    model = EmotionBiLSTM(
        input_size=input_dim,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.35
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
        min_lr=1e-6
    )

    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = f"best_model_{FEATURE_KEY}.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for feats, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * feats.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

        train_loss /= total
        train_acc = 100. * correct / total

        # Validation (using held-out test set as val)
        model.eval()
        val_loss = 0.0
        preds, trues = [], []

        with torch.no_grad():
            for feats, labels in test_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                outputs = model(feats)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * feats.size(0)
                _, pred = outputs.max(1)
                preds.extend(pred.cpu().numpy())
                trues.extend(labels.cpu().numpy())

        val_loss /= len(test_ds)
        val_acc = accuracy_score(trues, preds) * 100
        val_f1 = f1_score(trues, preds, average='weighted') * 100

        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"         | Val   Loss: {val_loss:.4f}  | Acc: {val_acc:.2f}% | F1: {val_f1:.2f}%")

        scheduler.step(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"   → Saved best model (val acc = {best_acc:.2f}%) at {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest validation accuracy: {best_acc:.2f}% at epoch {best_epoch}")

    # Final report on test set
    print("\nFinal Test Set Classification Report:")
    print(classification_report(trues, preds, target_names=le.classes_, digits=4))

    # Confusion matrix
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix – {FEATURE_KEY}')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{FEATURE_KEY}.png')
    plt.show()

    print(f"Model saved to: {best_model_path}")
    print(f"Confusion matrix saved to: confusion_matrix_{FEATURE_KEY}.png")


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_and_evaluate()