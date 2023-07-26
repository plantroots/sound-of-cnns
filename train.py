import os
import numpy as np

from keras.datasets import mnist
from vae import VAE
import librosa

LEARNING_RATE = 0.0005
BATCH_SIZE = 1
EPOCHS = 200

SPECTROGRAMS_PATH = r"c:\Dataset\filtered_kicks\sounds"


def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            # these are 2D while the greyscale images where 3D (28, 28, 1)
            # we need to add an extra dimensions to them
            signal, _ = librosa.load(file_path, mono=True, sr=44100)  # (n_bins, n_frames)
            if len(signal) < 76734:
                padding_to_add = 76734 - len(signal)
                signal = np.append(signal, np.zeros(padding_to_add))
            x_train.append(signal)
    x_train = np.array(x_train)
    # reshaping for audio data
    x_train = x_train[..., np.newaxis, np.newaxis]  # -> (3000, 76734, 1, 1)
    return x_train


def train(x_train, learning_rate, batch_size, epochs):
    # for MNIST
    # autoencoder = VAE(
    #     input_shape=(28, 28, 1),
    #     conv_filters=(32, 64, 64, 64),
    #     conv_kernels=(3, 3, 3, 3),
    #     conv_strides=(1, 2, 2, 1),
    #     latent_space_dim=2  # increase this for better fidelity
    # )
    # for AUDIO
    autoencoder = VAE(
        input_shape=(76734, 1, 1),
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
    # x_train, _, _, _ = load_mnist()
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    # autoencoder2 = Autoencoder.load("model")
    # autoencoder2.summary()
