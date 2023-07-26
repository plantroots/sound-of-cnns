import os
import librosa
import numpy as np

from vae import VAE

AUDIO_DIR = r"c:\Dataset\filtered_kicks\sounds"
# 22050/44100 -> 38368/76736
SAMPLE_RATE = 22050
NUM_OF_SAMPLES_IN_A_FILE = 38368  # 76736 instead of 76734 so that the graph works

LEARNING_RATE = 0.0005
BATCH_SIZE = 2
EPOCHS = 30


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            # these are 2D while the greyscale images where 3D (28, 28, 1)
            # we need to add an extra dimensions to them
            signal, _ = librosa.load(file_path, mono=True, sr=SAMPLE_RATE)  # (n_bins, n_frames)

            if len(signal) < NUM_OF_SAMPLES_IN_A_FILE:
                padding_to_add = NUM_OF_SAMPLES_IN_A_FILE - len(signal)
                signal = np.append(signal, np.zeros(padding_to_add))
            x_train.append(signal)
    x_train = np.array(x_train)
    # reshaping for audio data
    x_train = x_train[..., np.newaxis, np.newaxis]  # -> (2737, 76736, 1, 1)
    return x_train


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
    x_train = load_fsdd(AUDIO_DIR)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    # autoencoder2 = Autoencoder.load("model")
    # autoencoder2.summary()
