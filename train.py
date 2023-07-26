import os
import librosa
import numpy as np

from vae import VAE

AUDIO_DIR = r"c:\Dataset\filtered_kicks_small"
# 22050/44100 -> 38368/76736
SAMPLE_RATE = 22050
NUM_OF_SAMPLES_IN_A_FILE = 38368  # 76736 instead of 76734 so that the graph works

LEARNING_RATE = 0.0001
BATCH_SIZE = 2
EPOCHS = 1000


def load_dataset(audio_dir):
    train_set = []
    for root, _, file_names in os.walk(audio_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            signal, _ = librosa.load(file_path, mono=True, sr=SAMPLE_RATE)

            mean = np.mean(signal)
            std = np.std(signal)
            normalized_signal = (signal - mean) / std

            if len(normalized_signal) < NUM_OF_SAMPLES_IN_A_FILE:
                padding_to_add = NUM_OF_SAMPLES_IN_A_FILE - len(normalized_signal)
                normalized_signal = np.append(normalized_signal, np.zeros(padding_to_add))

            train_set.append(normalized_signal)

    train_set = np.array(train_set)
    train_set = train_set[..., np.newaxis, np.newaxis]  # -> (2737, 76736, 1, 1)
    return train_set


def train(x_train, learning_rate, batch_size, epochs):
    # for AUDIO
    autoencoder = VAE(
        input_shape=(NUM_OF_SAMPLES_IN_A_FILE, 1, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train = load_dataset(AUDIO_DIR)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    # autoencoder2 = Autoencoder.load("model")
    # autoencoder2.summary()
