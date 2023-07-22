import os
import numpy as np

from keras.datasets import mnist
from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 5

SPECTROGRAMS_PATH = r"C:\Datasets\FSDD\spectrograms"


# def load_mnist():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
#     # normalize
#     x_train = x_train.astype("float32") / 255
#     x_train = x_train.reshape(x_train.shape + (1,))
#
#     # add extra dimension - the channel
#     x_test = x_test.astype("float32") / 255
#     x_test = x_test.reshape(x_test.shape + (1,))
#
#     return x_train, y_train, x_test, y_test

def load_fsdd(spectrograms_path):
    x_train = []
    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            # these are 2D while the greyscale images where 3D (28, 28, 1)
            # we need to add an extra dimensions to them
            spectrogram = np.load(file_path)  # (n_bins, n_frames)
            x_train.append(spectrogram)
    x_train = np.array(x_train)
    # reshaping for audio data
    x_train = x_train[..., np.newaxis]  # -> (3000, 256, 64, 1)
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
        input_shape=(256, 64, 1),
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
    autoencoder = train(x_train[:100], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    # autoencoder2 = Autoencoder.load("model")
    # autoencoder2.summary()
