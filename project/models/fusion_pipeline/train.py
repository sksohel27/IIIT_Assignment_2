# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer

# Emotion mapping (from dataset folders)
EMOTIONS = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'ps': 5, 'sad': 6}
NUM_CLASSES = len(EMOTIONS)

# Custom Dataset
class TESSDataset(Dataset):
    def __init__(self, df, max_audio_len=160000):  # 10s at 16kHz
        self.df = df
        self.max_audio_len = max_audio_len
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Load audio
        audio, sr = librosa.load(row['file_path'], sr=16000)
        audio = librosa.effects.trim(audio)[0]  # Trim silence
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]
        else:
            audio = np.pad(audio, (0, self.max_audio_len - len(audio)))
        audio = torch.from_numpy(audio).float().unsqueeze(0)  # (1, time)
        mel = self.mel_transform(audio)  # (1, n_mels, time)

        # TF-IDF
        tfidf = torch.from_numpy(row['tfidf']).float()

        label = row['label']
        return mel.squeeze(0), tfidf, label

# Collate for batching
def collate_fn(batch):
    mels, tfidfs, labels = zip(*batch)
    mels = pad_sequence(mels, batch_first=True)  # Pad mels if needed, but since fixed len, maybe not
    tfidfs = torch.stack(tfidfs)
    labels = torch.tensor(labels, dtype=torch.long)
    return mels, tfidfs, labels

# Fusion Model
class FusionModel(nn.Module):
    def __init__(self, tfidf_dim, hidden_dim=256):
        super().__init__()
        # Speech branch: CNN + LSTM on Mel
        self.speech_cnn = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1)
        self.speech_pool = nn.MaxPool2d(2)
        self.speech_lstm = nn.LSTM(32 * 64, hidden_dim, bidirectional=True, batch_first=True)  # After pooling

        # Text branch: Linear on TF-IDF
        self.text_fc = nn.Linear(tfidf_dim, hidden_dim * 2)

        # Fusion & Classifier
        self.fc1 = nn.Linear(hidden_dim*2 * 2, 512)  # Speech (512) + Text (512) = 1024 -> 512
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, mel, text_tfidf):
        # Speech
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, time)
        speech = nn.ReLU()(self.speech_cnn(mel))
        speech = self.speech_pool(speech)  # (B, 32, n_mels/2, time/2)
        speech = speech.view(speech.size(0), speech.size(3), -1)  # (B, time/2, 32*(n_mels/2))
        speech, _ = self.speech_lstm(speech)
        speech = speech[:, -1, :]  # Last hidden (B, 512)

        # Text
        text = self.text_fc(text_tfidf)  # (B, 512)

        # Fusion
        fused = torch.cat([speech, text], dim=1)
        out = nn.ReLU()(self.fc1(fused))
        out = self.dropout(out)
        return self.fc2(out)

# Data Preparation
def prepare_data(data_dir='TESS_Dataset/TESS Toronto emotional speech set data'):
    files = []
    emotion_map = {'surprise': 'ps', 'surprised': 'ps'}
    for root, _, filenames in os.walk(data_dir):
        for f in filenames:
            if f.endswith('.wav'):
                path = os.path.join(root, f)
                emotion = os.path.basename(root).split('_')[-1].lower()
                emotion = emotion_map.get(emotion, emotion)
                if emotion not in EMOTIONS:
                    continue
                parts = f.split('_')
                if len(parts) < 3:
                    continue
                word = parts[1]
                text = f"Say the word {word}"
                label = EMOTIONS[emotion]
                files.append({'file_path': path, 'text': text, 'label': label})
    df = pd.DataFrame(files)
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'])
    return train_df, val_df, test_df

# Training Loop
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for mel, tfidfs, labels in loader:
        mel, tfidfs, labels = mel.to(device), tfidfs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(mel, tfidfs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# Validation
def validate(model, loader, criterion, device):
    model.eval()
    preds, trues = [], []
    total_loss = 0
    with torch.no_grad():
        for mel, tfidfs, labels in loader:
            mel, tfidfs, labels = mel.to(device), tfidfs.to(device), labels.to(device)
            outputs = model(mel, tfidfs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted')
    return total_loss / len(loader), acc, f1

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data directory (change to your path)
    data_dir = 'TESS_Dataset/TESS Toronto emotional speech set data'  # Update if needed
    train_df, val_df, test_df = prepare_data(data_dir)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    train_tfidf = vectorizer.fit_transform(train_df['text']).toarray()
    val_tfidf = vectorizer.transform(val_df['text']).toarray()
    test_tfidf = vectorizer.transform(test_df['text']).toarray()

    train_df['tfidf'] = list(train_tfidf)
    val_df['tfidf'] = list(val_tfidf)
    test_df['tfidf'] = list(test_tfidf)

    tfidf_dim = train_tfidf.shape[1]

    # Datasets and loaders
    train_ds = TESSDataset(train_df)
    val_ds = TESSDataset(val_df)
    test_ds = TESSDataset(test_df)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=32, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

    # Model
    model = FusionModel(tfidf_dim).to(device)

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    num_epochs = 10  # Adjust as needed
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val F1 {val_f1:.4f}')

    # Save model
    os.makedirs('Results', exist_ok=True)
    torch.save(model.state_dict(), 'Results/fusion_model.pth')

    # Test evaluation
    _, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')

    # Save accuracies to CSV for tables
    pd.DataFrame({'Model': ['Fusion'], 'Acc': [test_acc], 'F1': [test_f1]}).to_csv('Results/fusion_accuracy.csv', index=False)

    print("Training completed! Model saved to Results/fusion_model.pth")