import os
import pickle
import librosa
import numpy as np

from vae import VAE

# AUDIO_DIR = r"c:\Dataset\filtered_kicks"
AUDIO_DIR = r"c:\Dataset\filtered_kicks_small"

# GOOGLE COLAB PATH
# AUDIO_DIR = "/content/drive/MyDrive/Music/filtered_kicks"

METADATA_DIR = r"C:\Code\sound-of-cnns\crafting_the_dataset\metadata"

# 22050/44100 -> 38368/76736
SAMPLE_RATE = 22050
NUM_OF_SAMPLES_IN_A_FILE = 40960


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
            signal, _ = librosa.load(file_path, mono=True, sr=SAMPLE_RATE)

            mean = np.mean(signal)
            std = np.std(signal)
            max_value = np.max(np.abs(signal))

            means.append(mean)
            standard_deviations.append(std)
            max_values.append(max_value)

            # normalized_signal = (signal - mean) / std
            normalized_signal = peak_amplitude_normalization(signal)

            if len(normalized_signal) < NUM_OF_SAMPLES_IN_A_FILE:
                padding_to_add = NUM_OF_SAMPLES_IN_A_FILE - len(normalized_signal)
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
        with open(os.path.join(METADATA_DIR, "train_set_mean_stddev_and_max_values.pkl"), "wb") as f:
            pickle.dump(mean_and_stddev_of_the_train_dataset, f)

    return train_set


def average_list_elements(input_list):
    if not input_list:
        return None
    total_sum = sum(input_list)
    average = total_sum / len(input_list)
    return average


if __name__ == "__main__":

    # LEARNING_RATE = 0.00001
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 4
    EPOCHS = 2500
    TRAIN = True
    # TRAIN = False

    vae = VAE(
        input_shape=(NUM_OF_SAMPLES_IN_A_FILE, 1),
        conv_filters=(256, 256, 128, 64, 32),
        # conv_kernels=(10, 10, 20, 20, 20),
        conv_kernels=(16, 8, 20, 20, 40),
        conv_strides=(8, 4, 2, 2, 2),
        latent_space_dim=5
    )

    vae.summary()

    if TRAIN:
        vae.compile(LEARNING_RATE)

        try:
            x_train = load_dataset(AUDIO_DIR)
            vae.train(x_train, BATCH_SIZE, EPOCHS)
        except KeyboardInterrupt:
            print("\n--> saving model <--")

        vae.save("model")
