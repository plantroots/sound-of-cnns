import os
import wandb
import pickle
import librosa
import numpy as np

from vae import VAE

# LOCAL/COLAB PATHS
AUDIO_DIR = r"c:\Dataset\filtered_kicks"
# AUDIO_DIR = r"/content/drive/MyDrive/Music/filtered_kicks"

TRAIN = True
NOTEBOOK_RUN = False
NUMBER_OF_SAMPLES_IN_A_FILE = 40960
# sample rate: # 22050/44100 -> 38368/76736

VAE_ARCHITECTURE = {
    "input_shape": (NUMBER_OF_SAMPLES_IN_A_FILE, 1),
    "conv_filters": (256, 256, 128, 64, 32),
    "conv_kernels": (16, 8, 20, 20, 40),
    "conv_strides": (8, 4, 2, 2, 2),
    "latent_space_dim": 5
}

wandb.init(
    project="VAE_GENERATOR_TRAIN",
    config={
        "audio_dir": AUDIO_DIR,
        "metadata_dir": r"C:\Code\sound-of-cnns\crafting_the_dataset\metadata",
        "sample_rate": 22050,
        "number_of_samples_in_a_file": NUMBER_OF_SAMPLES_IN_A_FILE,
        "epochs": 10000,
        "batch_size": 16,
        "learning_rate": 0.001,
        "vae_architecture": VAE_ARCHITECTURE
    }
)

config = wandb.config


def peak_amplitude_normalization(audio_data, target_max=1.0):
    max_val = np.max(np.abs(audio_data))
    normalized_data = audio_data * (target_max / max_val)
    return normalized_data


def load_dataset(audio_dir, save_metadata=False):
    train_set = []
    means = []
    standard_deviations = []
    max_values = []
    for root, _, file_names in os.walk(audio_dir):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            signal, _ = librosa.load(file_path, mono=True, sr=config.sample_rate)

            mean = np.mean(signal)
            std = np.std(signal)
            max_value = np.max(np.abs(signal))

            means.append(mean)
            standard_deviations.append(std)
            max_values.append(max_value)

            # normalized_signal = (signal - mean) / std
            normalized_signal = peak_amplitude_normalization(signal)

            if len(normalized_signal) < config.number_of_samples_in_a_file:
                padding_to_add = config.number_of_samples_in_a_file - len(normalized_signal)
                normalized_signal = np.append(normalized_signal, np.zeros(padding_to_add))

            # array_reshaped = np.reshape(normalized_signal, (640, 64))
            train_set.append(normalized_signal)

    train_set = np.array(train_set)
    train_set = train_set[..., np.newaxis]

    mean_and_stddev_of_the_train_dataset = {
        "mean": average_list_elements(means),
        "stddev": average_list_elements(standard_deviations),
        "max_values": max_values
    }

    if save_metadata:
        with open(os.path.join(config.metadata_dir, "train_set_mean_stddev_and_max_values.pkl"), "wb") as f:
            pickle.dump(mean_and_stddev_of_the_train_dataset, f)

    return train_set


def average_list_elements(input_list):
    if not input_list:
        return None
    total_sum = sum(input_list)
    average = total_sum / len(input_list)
    return average


if __name__ == "__main__":

    vae = VAE(
        input_shape=VAE_ARCHITECTURE["input_shape"],
        conv_filters=VAE_ARCHITECTURE["conv_filters"],
        conv_kernels=VAE_ARCHITECTURE["conv_kernels"],
        conv_strides=VAE_ARCHITECTURE["conv_strides"],
        latent_space_dim=VAE_ARCHITECTURE["latent_space_dim"]
    )

    vae.summary()

    if TRAIN:
        vae.compile(config.learning_rate)

        try:
            x_train = load_dataset(config.audio_dir)
            vae.train(x_train, config.batch_size, config.epochs)
        except KeyboardInterrupt:
            print("\n--> saving model <--")

        vae.save("model")
        if NOTEBOOK_RUN:
            wandb.finish()
