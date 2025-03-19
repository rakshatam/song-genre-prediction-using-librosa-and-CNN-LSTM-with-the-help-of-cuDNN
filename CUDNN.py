import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GRU, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Path to the dataset
DATASET_PATH = "C:/Users/Lenovo/Downloads/archive (6)/Data/genres_original"

# Augmentation function
def augment_audio(audio, sample_rate):
    augmented = []
    # Time Stretch
    augmented.append(librosa.effects.time_stretch(audio, rate=1.2))
    augmented.append(librosa.effects.time_stretch(audio, rate=0.8))
    # Pitch Shift
    augmented.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=-2))
    return augmented

# Feature extraction function
def extract_features(file_path, n_mfcc=60, max_pad_len=130):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast', duration=30)
        features = []
        
        # Add base features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        
        # Normalize and stack features
        scaler = StandardScaler()
        for feature in [mfccs, chroma, spectral_contrast, mel_spectrogram, spectral_rolloff, zero_crossing_rate]:
            feature = scaler.fit_transform(feature.T).T
            features.append(feature)
        
        features = np.concatenate(features, axis=0)
        
        # Pad or trim
        pad_width = max_pad_len - features.shape[1]
        if pad_width > 0:
            features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            features = features[:, :max_pad_len]
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process dataset
def process_audio_files(dataset_path):
    data = []
    labels = []
    genres = os.listdir(dataset_path)
    
    for genre in genres:
        genre_folder = os.path.join(dataset_path, genre)
        if not os.path.isdir(genre_folder):
            continue
        
        for file in os.listdir(genre_folder):
            if file.endswith('.wav'):
                file_path = os.path.join(genre_folder, file)
                features = extract_features(file_path)
                if features is not None:
                    data.append(features)
                    labels.append(genre)
                    # Augment data
                    audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=30)
                    for augmented_audio in augment_audio(audio, sr):
                        augmented_features = extract_features(file_path)
                        if augmented_features is not None:
                            data.append(augmented_features)
                            labels.append(genre)
    
    return np.array(data), np.array(labels)

# Load and process the dataset
X, y = process_audio_files(DATASET_PATH)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=len(np.unique(y)))

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, stratify=y_encoded, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weights = dict(enumerate(class_weights))

# Build the optimized RNN model
model = Sequential([
    Conv1D(64, kernel_size=3, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    BatchNormalization(),
    Dropout(0.3),
    
    GRU(256, activation='tanh', return_sequences=True),
    Dropout(0.4),
    GRU(128, activation='tanh'),
    Dropout(0.4),
    
    Dense(128, activation='tanh'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30, batch_size=16,
    callbacks=[early_stopping, lr_scheduler],
    class_weight=class_weights
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot Confusion Matrix
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Predict for a new song
def predict_genre_for_new_song(file_path, model, label_encoder, n_mfcc=60, max_pad_len=130):
    # Extract features for the new song
    features = extract_features(file_path, n_mfcc=n_mfcc, max_pad_len=max_pad_len)
    if features is None:
        print("Error: Could not extract features from the audio file.")
        return
    
    # Reshape to match model input
    features = features.reshape(1, features.shape[0], features.shape[1])
    prediction = model.predict(features)
    
    # Predicted genre
    predicted_genre = label_encoder.inverse_transform([np.argmax(prediction)])
    
    # Display probabilities for each genre
    print("Prediction probabilities:")
    for genre, prob in zip(label_encoder.classes_, prediction[0]):
        print(f"{genre}: {prob:.2f}")
    
    return predicted_genre[0]

# Example usage for genre prediction
new_song_path = "C:/Users/Lenovo/Desktop/029813.mp3"  # Replace with your new song file path
predicted_genre = predict_genre_for_new_song(new_song_path, model, label_encoder)
print(f"Predicted Genre: {predicted_genre}")
