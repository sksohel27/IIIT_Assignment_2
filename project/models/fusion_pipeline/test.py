# test.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from torch.nn.utils.rnn import pad_sequence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Emotion mapping (from dataset folders)
EMOTIONS = {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'ps': 5, 'sad': 6}
NUM_CLASSES = len(EMOTIONS)
EMOTION_LABELS = list(EMOTIONS.keys())

# Custom Dataset (same as train)
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
    mels = pad_sequence(mels, batch_first=True)
    tfidfs = torch.stack(tfidfs)
    labels = torch.tensor(labels, dtype=torch.long)
    return mels, tfidfs, labels

# Fusion Model (modified to return representations for t-SNE)
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

    def forward(self, mel, text_tfidf, return_reps=False):
        # Speech
        mel = mel.unsqueeze(1)  # (B, 1, n_mels, time)
        speech = nn.ReLU()(self.speech_cnn(mel))
        speech = self.speech_pool(speech)  # (B, 32, n_mels/2, time/2)
        speech = speech.view(speech.size(0), speech.size(3), -1)  # (B, time/2, 32*(n_mels/2))
        speech, _ = self.speech_lstm(speech)
        speech_rep = speech[:, -1, :]  # Last hidden (B, 512)

        # Text
        text_rep = self.text_fc(text_tfidf)  # (B, 512)

        # Fusion
        fused_rep = torch.cat([speech_rep, text_rep], dim=1)
        out = nn.ReLU()(self.fc1(fused_rep))
        out = self.dropout(out)
        logits = self.fc2(out)

        if return_reps:
            return logits, speech_rep, text_rep, fused_rep
        return logits

# Data Preparation (for test)
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
    return test_df  # Return test split for evaluation

# Function to extract representations
def extract_representations(model, loader, device):
    model.eval()
    speech_reps = []
    text_reps = []
    fused_reps = []
    labels_list = []
    with torch.no_grad():
        for mel, tfidfs, labels in loader:
            mel, tfidfs = mel.to(device), tfidfs.to(device)
            _, speech_rep, text_rep, fused_rep = model(mel, tfidfs, return_reps=True)
            speech_reps.append(speech_rep.cpu().numpy())
            text_reps.append(text_rep.cpu().numpy())
            fused_reps.append(fused_rep.cpu().numpy())
            labels_list.append(labels.numpy())
    return (np.concatenate(speech_reps), np.concatenate(text_reps),
            np.concatenate(fused_reps), np.concatenate(labels_list))

# Function to plot t-SNE
def plot_tsne(reps, labels, title, file_path):
    tsne = TSNE(n_components=2, random_state=42)
    reps_2d = tsne.fit_transform(reps)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reps_2d[:, 0], reps_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=range(NUM_CLASSES), label='Emotions')
    plt.clim(-0.5, NUM_CLASSES - 0.5)
    plt.title(title)
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(file_path)
    plt.close()

# Function to get predictions
def get_predictions(model, loader, device):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for mel, tfidfs, labels in loader:
            mel, tfidfs, labels = mel.to(device), tfidfs.to(device), labels.to(device)
            outputs = model(mel, tfidfs)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(labels.cpu().numpy())
    return np.array(trues), np.array(preds)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data directory (change to your path)
    data_dir = 'TESS_Dataset/TESS Toronto emotional speech set data'  # Update if needed
    test_df = prepare_data(data_dir)

    # TF-IDF Vectorizer (fit on train would be ideal, but using test for demo; in practice, load from train)
    vectorizer = TfidfVectorizer()
    test_tfidf = vectorizer.fit_transform(test_df['text']).toarray()  # Note: In real use, use fitted from train
    test_df['tfidf'] = list(test_tfidf)

    tfidf_dim = test_tfidf.shape[1]

    # Test dataset and loader
    test_ds = TESSDataset(test_df)
    test_loader = DataLoader(test_ds, batch_size=32, collate_fn=collate_fn)

    # Load model
    model = FusionModel(tfidf_dim).to(device)
    model.load_state_dict(torch.load('Results/fusion_model.pth', weights_only=True))

    # Extract representations for t-SNE
    speech_reps, text_reps, fused_reps, labels = extract_representations(model, test_loader, device)

    # Create results directory
    os.makedirs('Results/plots', exist_ok=True)

    # Plot t-SNE for each
    plot_tsne(speech_reps, labels, 't-SNE of Temporal Modeling (Speech) Representations', 'Results/plots/speech_tsne.png')
    plot_tsne(text_reps, labels, 't-SNE of Contextual Modeling (Text) Representations', 'Results/plots/text_tsne.png')
    plot_tsne(fused_reps, labels, 't-SNE of Fusion Representations', 'Results/plots/fusion_tsne.png')

    print("Visualizations saved to Results/plots/")

    # Get predictions for confusion matrix
    true_labels, pred_labels = get_predictions(model, test_loader, device)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)

    # Plot and save
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Fusion Model')
    plt.savefig('Results/plots/confusion_matrix.png')
    plt.close()

    # Compute metrics
    test_acc = accuracy_score(true_labels, pred_labels)
    test_f1 = f1_score(true_labels, pred_labels, average='weighted')

    print(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')
    print("Confusion matrix saved to Results/plots/confusion_matrix.png")
    print("Confusion Matrix:\n", cm)