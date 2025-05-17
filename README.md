# Music Genre Classification using Convolutional Recurrent Neural Networks (CRNN)

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow / Keras](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Librosa](https://img.shields.io/badge/Librosa-%23282C34.svg?style=flat&logo=librosa&logoColor=white)](https://librosa.org/)

This project implements a music genre classification system using a Convolutional Recurrent Neural Network (CRNN) in TensorFlow/Keras. It processes raw audio files, extracts a comprehensive set of acoustic features, applies data augmentation techniques, and trains a robust model to categorize music into distinct genres. The pipeline includes feature engineering, model training with callbacks and class weighting, and thorough evaluation using a confusion matrix and classification report.

---

## Features

* **Comprehensive Feature Extraction:** Utilizes `librosa` to extract MFCCs, Chroma STFT, Spectral Contrast, Mel Spectrogram, Spectral Rolloff, and Zero Crossing Rate.
* **Data Augmentation:** Enhances dataset diversity and model robustness through time stretching and pitch shifting of audio samples.
* **CRNN Model Architecture:** Employs a combination of 1D Convolutional layers for local pattern recognition and GRU (Gated Recurrent Unit) layers for capturing temporal dependencies in audio features.
* **Class Weighting:** Addresses potential class imbalance in the dataset during training to improve fairness and accuracy across all genres.
* **Optimized Training:** Integrates Keras Callbacks like `EarlyStopping` (for preventing overfitting) and `ReduceLROnPlateau` (for dynamic learning rate adjustment).
* **Detailed Evaluation:** Provides comprehensive evaluation metrics including overall accuracy, a visual confusion matrix, and a detailed classification report (precision, recall, f1-score per class).
* **New Song Prediction:** Includes a function to predict the genre of a new, unseen audio file.

---

## Technical Deep Dive: Pipeline Overview

The project's workflow is segmented into several key stages:

### 1. Data Acquisition and Preprocessing

* **Dataset Loading:** The system expects a dataset organized by genre (e.g., the GTZAN dataset) where each genre is a separate folder containing `.wav` audio files.
* **Feature Extraction (`extract_features`):** For each audio file, `librosa` is used to compute:
    * **Mel-frequency Cepstral Coefficients (MFCCs):** Represent the short-term power spectrum of a sound.
    * **Chroma Feature:** Captures the twelve different pitch classes (C, C#, D, etc.).
    * **Spectral Contrast:** Measures the contrast between spectral peaks and valleys.
    * **Mel Spectrogram:** A time-frequency representation of the audio, useful for capturing timbre.
    * **Spectral Rolloff:** Indicates the shape of the spectral envelope.
    * **Zero Crossing Rate:** Measures the rate at which the signal changes sign.
    * All features are standardized using `StandardScaler` and then concatenated.
    * Features are then padded or trimmed to a fixed `max_pad_len` (130 frames by default) to ensure uniform input dimensions for the neural network.
* **Data Augmentation (`augment_audio`):** To increase the size and diversity of the training data and improve model generalization, each original audio file is augmented by:
    * **Time Stretching:** Changing the speed of the audio without altering its pitch.
    * **Pitch Shifting:** Changing the pitch of the audio without altering its speed.
    * Features are extracted from these augmented samples as well.

### 2. Model Architecture (CRNN)

The neural network is a Sequential Keras model designed to leverage the strengths of both Convolutional and Recurrent layers:

* **`Conv1D` Layers:** Capture local patterns and features within the time-series audio data.
* **`MaxPooling1D`:** Reduces dimensionality and captures the most salient features.
* **`BatchNormalization`:** Stabilizes and speeds up training.
* **`GRU` Layers:** Gated Recurrent Units are used to process the sequential nature of the features, capturing long-term dependencies in the audio. Two GRU layers are stacked for deeper temporal learning.
* **`Dropout` Layers:** Applied after convolutional and recurrent layers to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.
* **`Dense` Layers:** Fully connected layers for high-level feature combination.
* **`Softmax` Output Layer:** Outputs probabilities for each music genre.

### 3. Training Strategy

* **Optimizer:** Adam optimizer is used for its efficiency and adaptive learning rate.
* **Loss Function:** `categorical_crossentropy` is chosen as the loss function, suitable for multi-class classification.
* **Class Weights:** `compute_class_weight` is used to calculate and apply balanced class weights during training. This is crucial for datasets where some genres might have significantly more or fewer samples than others, preventing the model from being biased towards majority classes.
* **Callbacks:**
    * `EarlyStopping`: Monitors validation loss and stops training if it doesn't improve for a specified number of epochs (`patience=10`), restoring the best model weights.
    * `ReduceLROnPlateau`: Reduces the learning rate by a factor (0.5 by default) if the validation loss plateaus (`patience=5`), helping the model converge better.

### 4. Evaluation and Prediction

* After training, the model's performance is evaluated on the test set.
* A **confusion matrix** is generated and visualized using `matplotlib` and `seaborn` to show correct and incorrect predictions for each genre.
* A detailed **classification report** provides precision, recall, and F1-score for each genre, offering a more granular view of the model's performance beyond just overall accuracy.
* A utility function `predict_genre_for_new_song` allows users to test the trained model on new audio files.

---
