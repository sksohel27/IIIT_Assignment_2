import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

FEATURE_KEY = 'rich_feature_sequence'   # Must match what was used in training
MODEL_PATH = f"best_model_{FEATURE_KEY}.pth"

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


# ── BiLSTM with Attention (same as in train.py) ───────────────────────────
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

        lstm_out_dim = hidden_size * 2

        self.attention = nn.Linear(lstm_out_dim, 1)

        concat_dim = lstm_out_dim * 4

        self.fc = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)                      
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)     
        attn_context = torch.sum(lstm_out * attn_w, dim=1)          

        mean_pool  = lstm_out.mean(dim=1)
        max_pool, _ = lstm_out.max(dim=1)
        last_hidden = lstm_out[:, -1, :]

        combined = torch.cat([mean_pool, max_pool, last_hidden, attn_context], dim=1)
        return self.fc(combined)


# ── Evaluation ────────────────────────────────────────────────────────────
def evaluate_model():
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Using feature: {FEATURE_KEY}")
    print(f"Device: {DEVICE}\n")

    # Load data
    test_df = pd.read_pickle("test_speaker_split.pkl")

    # Label encoder
    le = LabelEncoder()
    le.fit(EMOTION_LABELS)

    test_ds = TessSequenceDataset(test_df, FEATURE_KEY, le)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Test samples: {len(test_ds)}")
    print(f"Classes: {list(le.classes_)}\n")

    # Model
    input_dim = test_ds[0][0].shape[-1]
    model = EmotionBiLSTM(
        input_size=input_dim,
        hidden_size=128,
        num_layers=2,
        num_classes=NUM_CLASSES,
        dropout=0.35
    ).to(DEVICE)

    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Model loaded successfully!\n")

    # Inference
    preds, trues = [], []
    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(feats)
            _, pred = outputs.max(1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(trues, preds) * 100
    f1 = f1_score(trues, preds, average='weighted') * 100

    print(f"Test Accuracy: {acc:.2f}%")
    print(f"Test F1-Score (weighted): {f1:.2f}%\n")

    print("Classification Report:")
    print(classification_report(trues, preds, target_names=le.classes_, digits=4))

    # Confusion matrix
    cm = confusion_matrix(trues, preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Confusion Matrix – {FEATURE_KEY} (Test Set)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_test_{FEATURE_KEY}.png')
    plt.show()

    print(f"Confusion matrix saved to: confusion_matrix_test_{FEATURE_KEY}.png")


# ── Run ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_model()