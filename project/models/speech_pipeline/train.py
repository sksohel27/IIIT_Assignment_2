# train.py
# Speech Emotion Recognition Training Pipeline
# This script handles: data loading, preprocessing, feature extraction, model training, and saving artifacts

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.utils import to_categorical

print("✓ All libraries imported successfully!")

# Dataset paths (adjust if running outside Colab)
DATA_DIR = "/content/TESS_Dataset"
TRAIN_DIR = "/content/TESS_Dataset/train"
TEST_DIR = "/content/TESS_Dataset/test"
CACHE_DIR = "/content/cached_features"
PREPROCESSED_DIR = "/content/preprocessed_data"
MODEL_DIR = "/content/models/speech_pipeline"

# Create directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Train path: {TRAIN_DIR}")
print(f"Test path: {TEST_DIR}")
print(f"Cache path: {CACHE_DIR}")
print(f"Preprocessed path: {PREPROCESSED_DIR}")
print(f"Model path: {MODEL_DIR}")

class SpeechPreprocessor:
    def __init__(
        self,
        sr: int = 22050,
        duration: float = 3.0,
        trim_silence: bool = True,
        top_db: int = 20,
        normalize: bool = True
    ):
        self.sr = sr
        self.duration = duration
        self.trim_silence = trim_silence
        self.top_db = top_db
        self.normalize = normalize
        self.max_length = int(sr * duration)

    def load_audio(self, file_path: str) -> np.ndarray:
        """Load audio file with specified sampling rate"""
        audio, _ = librosa.load(file_path, sr=self.sr, mono=True)
        return audio

    def trim_audio_silence(self, audio: np.ndarray) -> np.ndarray:
        """Trim leading and trailing silence from audio"""
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=self.top_db)
        return trimmed_audio

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        if np.max(np.abs(audio)) > 0:
            normalized_audio = audio / np.max(np.abs(audio))
        else:
            normalized_audio = audio
        return normalized_audio

    def pad_or_truncate(self, audio: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to fixed length"""
        if len(audio) > self.max_length:
            # Truncate from center
            start = (len(audio) - self.max_length) // 2
            fixed_audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            # Pad with zeros
            pad_width = self.max_length - len(audio)
            fixed_audio = np.pad(audio, (0, pad_width), mode='constant')
        else:
            fixed_audio = audio
        return fixed_audio

    def preprocess(self, file_path: str) -> np.ndarray:
        """Complete preprocessing pipeline"""
        # Load audio
        audio = self.load_audio(file_path)

        # Trim silence if enabled
        if self.trim_silence:
            audio = self.trim_audio_silence(audio)

        # Normalize if enabled
        if self.normalize:
            audio = self.normalize_audio(audio)

        # Pad or truncate to fixed length
        audio = self.pad_or_truncate(audio)

        return audio

    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 40,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """Extract MFCC features from audio"""
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length
        )
        return mfcc

    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """Extract Mel spectrogram from audio"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_spectral_features(
        self,
        audio: np.ndarray,
        n_fft: int = 2048,
        hop_length: int = 512
    ) -> Dict[str, np.ndarray]:
        """Extract various spectral features"""
        features = {}

        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )

        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )

        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=n_fft, hop_length=hop_length
        )

        features['chroma_stft'] = librosa.feature.chroma_stft(
            y=audio, sr=self.sr, n_fft=n_fft, hop_length=hop_length
        )

        features['rms'] = librosa.feature.rms(
            y=audio, frame_length=n_fft, hop_length=hop_length
        )

        return features

    def extract_all_features(
        self,
        audio: np.ndarray,
        include_mfcc: bool = True,
        include_mel: bool = True,
        include_spectral: bool = True
    ) -> Dict[str, np.ndarray]:
        """Extract all available features from audio"""
        all_features = {}

        if include_mfcc:
            all_features['mfcc'] = self.extract_mfcc(audio)
            # Add delta and delta-delta MFCCs
            all_features['mfcc_delta'] = librosa.feature.delta(all_features['mfcc'])
            all_features['mfcc_delta2'] = librosa.feature.delta(all_features['mfcc'], order=2)

        if include_mel:
            all_features['mel_spectrogram'] = self.extract_mel_spectrogram(audio)

        if include_spectral:
            spectral_feats = self.extract_spectral_features(audio)
            all_features.update(spectral_feats)

        return all_features

print("✓ SpeechPreprocessor class defined!")

class TESSDataLoader:
    def __init__(
        self,
        data_dir: str,
        preprocessor: SpeechPreprocessor,
        cache_dir: Optional[str] = None
    ):
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.label_encoder = LabelEncoder()

        self.emotion_map = {
            'angry': 'angry',
            'disgust': 'disgust',
            'fear': 'fear',
            'happy': 'happy',
            'neutral': 'neutral',
            'sad': 'sad',
            'ps': 'pleasant_surprise',
            'pleasant surprise': 'pleasant_surprise',
            'surprise': 'pleasant_surprise',
            'surprised': 'pleasant_surprise'
        }

    def _extract_emotion_from_folder(self, folder_name: str) -> str:
        """Extract standardized emotion label from folder name"""
        folder_name = folder_name.lower().replace('oaf_', '').replace('yaf_', '').replace('_', ' ')

        for key, value in self.emotion_map.items():
            if key in folder_name:
                return value

        return folder_name.strip()

    def get_file_paths_and_labels(self, split: str = 'train') -> Tuple[List[str], List[str]]:
        """Get all audio file paths and corresponding emotion labels"""
        split_dir = self.data_dir / split
        file_paths = []
        labels = []

        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")

        for emotion_folder in split_dir.iterdir():
            if not emotion_folder.is_dir():
                continue

            emotion = self._extract_emotion_from_folder(emotion_folder.name)

            for wav_file in emotion_folder.glob("*.wav"):
                file_paths.append(str(wav_file))
                labels.append(emotion)

        return file_paths, labels

    def load_and_preprocess_audio(
        self,
        file_path: str,
        feature_type: str = 'mfcc'
    ) -> np.ndarray:
        """Load and preprocess a single audio file"""
        audio = self.preprocessor.preprocess(file_path)

        if feature_type == 'mfcc':
            features = self.preprocessor.extract_mfcc(audio)
        elif feature_type == 'mel':
            features = self.preprocessor.extract_mel_spectrogram(audio)
        elif feature_type == 'all':
            all_features = self.preprocessor.extract_all_features(audio)
            features = np.vstack([
                all_features['mfcc'],
                all_features['mfcc_delta'],
                all_features['mfcc_delta2']
            ])
        else:
            raise ValueError(f"Invalid feature_type: {feature_type}")

        return features

    def load_dataset(
        self,
        split: str = 'train',
        feature_type: str = 'mfcc',
        use_cache: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess entire dataset"""
        cache_file = None
        if self.cache_dir and use_cache:
            cache_file = self.cache_dir / f"{split}_{feature_type}_features.pkl"

            if cache_file.exists():
                if verbose:
                    print(f"Loading cached features from {cache_file}")
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                return cache_data['X'], cache_data['y'], cache_data['y_encoded']

        file_paths, labels = self.get_file_paths_and_labels(split)

        if verbose:
            print(f"\nLoading {split} dataset:")
            print(f"  Total files: {len(file_paths)}")
            print(f"  Feature type: {feature_type}")

        features_list = []
        valid_labels = []

        iterator = tqdm(file_paths, desc=f"Processing {split} data") if verbose else file_paths

        for file_path, label in zip(iterator, labels):
            try:
                features = self.load_and_preprocess_audio(file_path, feature_type)
                features_list.append(features)
                valid_labels.append(label)
            except Exception as e:
                if verbose:
                    print(f"\nError processing {file_path}: {str(e)}")
                continue

        X = np.array(features_list)
        y = np.array(valid_labels)

        # Transpose to (n_samples, time_steps, n_features)
        X = X.transpose(0, 2, 1)

        # Encode labels
        if split == 'train':
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = self.label_encoder.transform(y)

        if verbose:
            print(f"\nDataset loaded:")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            print(f"  Number of classes: {len(np.unique(y))}")
            print(f"  Class distribution:")
            unique, counts = np.unique(y, return_counts=True)
            for emo, count in zip(unique, counts):
                print(f"    {emo:20s}: {count:4d} samples")

        if cache_file:
            if verbose:
                print(f"\nCaching features to {cache_file}")
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'X': X,
                    'y': y,
                    'y_encoded': y_encoded
                }, f)

        return X, y, y_encoded

    def get_class_names(self) -> List[str]:
        """Get list of emotion class names"""
        return list(self.label_encoder.classes_)

    def get_num_classes(self) -> int:
        """Get number of emotion classes"""
        return len(self.label_encoder.classes_)

    def create_validation_split(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        val_size: float = 0.15,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split training data into train and validation sets"""
        return train_test_split(
            X_train,
            y_train,
            test_size=val_size,
            random_state=random_state,
            stratify=y_train
        )

print("✓ TESSDataLoader class defined!")

# Initialize preprocessor
preprocessor = SpeechPreprocessor(
    sr=22050,
    duration=3.0,
    trim_silence=True,
    top_db=20,
    normalize=True
)

print("Preprocessor Configuration:")
print(f"  Sampling Rate: {preprocessor.sr} Hz")
print(f"  Duration: {preprocessor.duration} seconds")
print(f"  Max Length: {preprocessor.max_length} samples")
print(f"  Trim Silence: {preprocessor.trim_silence}")
print(f"  Normalize: {preprocessor.normalize}")

# Initialize data loader
data_loader = TESSDataLoader(
    data_dir=DATA_DIR,
    preprocessor=preprocessor,
    cache_dir=CACHE_DIR
)

print("✓ Data loader initialized!")

# Load training dataset
X_train, y_train, y_train_encoded = data_loader.load_dataset(
    split='train',
    feature_type='mfcc',
    use_cache=True,
    verbose=True
)

print("\n" + "="*70)
print("Training data loaded successfully!")
print("="*70)

# Load test dataset
X_test, y_test, y_test_encoded = data_loader.load_dataset(
    split='test',
    feature_type='mfcc',
    use_cache=True,
    verbose=True
)

print("\n" + "="*70)
print("Test data loaded successfully!")
print("="*70)

# Create validation split
X_train_final, X_val, y_train_final, y_val = data_loader.create_validation_split(
    X_train,
    y_train_encoded,
    val_size=0.15,
    random_state=42
)

print("\nData splits created:")
print(f"  Train: X={X_train_final.shape}, y={y_train_final.shape}")
print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
print(f"  Test:  X={X_test.shape}, y={y_test_encoded.shape}")

print(f"\nClass names: {data_loader.get_class_names()}")
print(f"Number of classes: {data_loader.get_num_classes()}")

# Plot class distribution
unique, counts = np.unique(y_train, return_counts=True)

plt.figure(figsize=(12, 6))
bars = plt.bar(unique, counts, color='steelblue', edgecolor='black', alpha=0.7)

for bar, count in zip(bars, counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}',
            ha='center', va='bottom', fontweight='bold')

plt.xlabel('Emotion', fontsize=12, fontweight='bold')
plt.ylabel('Number of Samples', fontsize=12, fontweight='bold')
plt.title('Training Set - Class Distribution', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f"{MODEL_DIR}/class_distribution.png", dpi=150, bbox_inches='tight')
plt.show()

print("\nClass Distribution:")
for emo, count in zip(unique, counts):
    print(f"  {emo:20s}: {count:4d} samples ({count/len(y_train)*100:.1f}%)")

# Save preprocessed data
np.save(f"{PREPROCESSED_DIR}/X_train.npy", X_train_final)
np.save(f"{PREPROCESSED_DIR}/y_train.npy", y_train_final)
np.save(f"{PREPROCESSED_DIR}/X_val.npy", X_val)
np.save(f"{PREPROCESSED_DIR}/y_val.npy", y_val)
np.save(f"{PREPROCESSED_DIR}/X_test.npy", X_test)
np.save(f"{PREPROCESSED_DIR}/y_test.npy", y_test_encoded)

# Save label encoder
with open(f"{PREPROCESSED_DIR}/label_encoder.pkl", 'wb') as f:
    pickle.dump(data_loader.label_encoder, f)

print("✓ Preprocessed data saved to:", PREPROCESSED_DIR)
print("\nFiles saved:")
for file in os.listdir(PREPROCESSED_DIR):
    file_path = os.path.join(PREPROCESSED_DIR, file)
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  {file:25s} ({size_mb:.2f} MB)")

print("="*70)
print("PREPROCESSING PIPELINE COMPLETED!")
print("="*70)

# ==================== MODEL TRAINING SECTION ====================

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
print(f"✓ Random seed set to {RANDOM_SEED}")

# Load preprocessed data
data_dir = PREPROCESSED_DIR

X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")
X_val = np.load(f"{data_dir}/X_val.npy")
y_val = np.load(f"{data_dir}/y_val.npy")
X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

with open(f"{data_dir}/label_encoder.pkl", 'rb') as f:
    label_encoder = pickle.load(f)

print("Data loaded successfully!")
print(f"\nData shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  y_val: {y_val.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Data reshaped for CNN:")
print(f"  X_train: {X_train.shape}")
print(f"  X_val: {X_val.shape}")
print(f"  X_test: {X_test.shape}")

# One-hot encode labels
num_classes = len(label_encoder.classes_)
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

print(f"\nLabels converted to categorical:")
print(f"  y_train: {y_train_cat.shape}")
print(f"  y_val: {y_val_cat.shape}")
print(f"  y_test: {y_test_cat.shape}")

# Normalize features
print("Normalizing MFCC features (using train statistics)...")
mean = np.mean(X_train, axis=(0, 1), keepdims=True)
std = np.std(X_train, axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

print(f"After normalization → Mean: {X_train.mean():.4f}, Std: {X_train.std():.4f}")

# Model hyperparameters
INPUT_SHAPE = X_train.shape[1:]
NUM_CLASSES = num_classes

CNN_FILTERS = [32, 64, 128]
CNN_KERNEL_SIZE = (3, 3)
POOL_SIZE = (2, 2)

LSTM_UNITS = [128, 64]
DROPOUT_RATE = 0.1

DENSE_UNITS = [128, 64]

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

print("Model Configuration:")
print(f"  Input Shape: {INPUT_SHAPE}")
print(f"  Number of Classes: {NUM_CLASSES}")
print(f"  CNN Filters: {CNN_FILTERS}")
print(f"  LSTM Units: {LSTM_UNITS}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")

def build_cnn_lstm_model(input_shape, num_classes):
    model = models.Sequential(name='CNN_BiLSTM_Hybrid')

    # CNN Block 1
    model.add(layers.Conv2D(
        filters=CNN_FILTERS[0],
        kernel_size=CNN_KERNEL_SIZE,
        activation='relu',
        padding='same',
        input_shape=input_shape
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=POOL_SIZE))
    model.add(layers.Dropout(DROPOUT_RATE))

    # CNN Block 2
    model.add(layers.Conv2D(
        filters=CNN_FILTERS[1],
        kernel_size=CNN_KERNEL_SIZE,
        activation='relu',
        padding='same'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=POOL_SIZE))
    model.add(layers.Dropout(DROPOUT_RATE))

    # CNN Block 3
    model.add(layers.Conv2D(
        filters=CNN_FILTERS[2],
        kernel_size=CNN_KERNEL_SIZE,
        activation='relu',
        padding='same'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(layers.Dropout(DROPOUT_RATE))

    # Reshape for LSTM
    model.add(layers.Reshape((-1, CNN_FILTERS[2]), name='reshape_for_lstm'))

    # BiLSTM Block
    model.add(layers.Bidirectional(
        layers.LSTM(LSTM_UNITS[0], return_sequences=True, dropout=DROPOUT_RATE),
        name='bi_lstm_1'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Bidirectional(
        layers.LSTM(LSTM_UNITS[1], return_sequences=False, dropout=DROPOUT_RATE),
        name='bi_lstm_2'
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    # Dense Classifier
    model.add(layers.Dense(DENSE_UNITS[0], activation='relu'))
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(DENSE_UNITS[1], activation='relu'))
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Clear session
tf.keras.backend.clear_session()

model = build_cnn_lstm_model(INPUT_SHAPE, NUM_CLASSES)

print("✓ CNN + BiLSTM model built successfully!")
model.summary()

# Count parameters
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
print(f"\nTotal parameters: {trainable_params + non_trainable_params:,} ({trainable_params:,} trainable)")

# Compile
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model compiled successfully!")

# Callbacks
callback_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath=f"{MODEL_DIR}/best_model.h5",
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    callbacks.TensorBoard(
        log_dir=f"{MODEL_DIR}/logs",
        histogram_freq=1
    )
]

print("✓ Callbacks configured!")

# Train the model
print("="*70)
print("STARTING MODEL TRAINING")
print("="*70)

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callback_list,
    verbose=1
)

print("\n" + "="*70)
print("✓ Training completed!")
print("="*70)

def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/training_history.png", dpi=150, bbox_inches='tight')
    plt.show()

    best_epoch = np.argmax(history.history['val_accuracy'])
    print(f"\nBest Validation Accuracy: {history.history['val_accuracy'][best_epoch]:.4f} (Epoch {best_epoch + 1})")

plot_training_history(history)

# Save final model
model.save(f"{MODEL_DIR}/final_model.h5")
print(f"✓ Final model saved to: {MODEL_DIR}/final_model.h5")

# Save training history
with open(f"{MODEL_DIR}/training_history.pkl", 'wb') as f:
    pickle.dump(history.history, f)
print(f"✓ Training history saved to: {MODEL_DIR}/training_history.pkl")

# Save results summary (basic)
results_summary = {
    'final_val_accuracy': float(max(history.history['val_accuracy'])),
    'final_val_loss': float(min(history.history['val_loss'])),
    'epochs_trained': len(history.history['loss']),
    'model_config': {
        'input_shape': list(INPUT_SHAPE),
        'num_classes': NUM_CLASSES,
        'cnn_filters': CNN_FILTERS,
        'lstm_units': LSTM_UNITS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE
    }
}

import json
with open(f"{MODEL_DIR}/results_summary.json", 'w') as f:
    json.dump(results_summary, f, indent=4)
print(f"✓ Results summary saved to: {MODEL_DIR}/results_summary.json")

# Optional cleanup: move folders
import shutil

base_path = "/content"
important_folder = os.path.join(base_path, "important_folders")
os.makedirs(important_folder, exist_ok=True)

folders_to_move = ["cached_features", "models", "preprocessed_data", "sample_data"]

for folder in folders_to_move:
    src = os.path.join(base_path, folder)
    dst = os.path.join(important_folder, folder)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved: {folder}")
    else:
        print(f"Not found: {folder}")

print("✅ Train.py completed! Model and data saved.")
print(f"Model directory: {MODEL_DIR}")