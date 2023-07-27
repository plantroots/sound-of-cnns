import os
import pickle
import librosa
import soundfile
import numpy as np

from train import average_list_elements

AUDIO_SIGNAL_PATH = r"c:\Dataset\filtered_kicks"
METADATA_DIR = r"C:\Code\sound-of-cnns\crafting_the_dataset\metadata"

NUM_OF_SAMPLES_IN_A_FILE = 40960
SAMPLE_RATE = 22050


def peak_amplitude_normalization(audio_data, target_max=1.0):
    max_val = np.max(np.abs(audio_data))
    normalized_data = audio_data * (target_max / max_val)
    return normalized_data


# TODO: make a list of all the original max value to try and extrapolate that to the generated sounds for reverting
def revert_peak_amplitude_normalization(normalized_data, original_max):
    # Get the maximum absolute amplitude from the normalized data
    max_val = np.max(np.abs(normalized_data))

    # Calculate the scaling factor used during normalization
    scaling_factor = original_max / max_val

    # Revert the normalization by multiplying with the scaling factor
    original_data = normalized_data * scaling_factor

    return original_data


def load_dataset_check(audio_dir):
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

            normalized_signal = peak_amplitude_normalization(signal)

            if len(normalized_signal) < NUM_OF_SAMPLES_IN_A_FILE:
                padding_to_add = NUM_OF_SAMPLES_IN_A_FILE - len(normalized_signal)
                normalized_signal = np.append(normalized_signal, np.zeros(padding_to_add))

            array_reshaped = np.reshape(normalized_signal, (640, 64))
            train_set.append(array_reshaped)

            break
        break
    train_set = np.array(train_set)
    train_set = train_set[..., np.newaxis]  # -> (2737, 76736, 1, 1)

    return train_set


x = load_dataset_check(AUDIO_SIGNAL_PATH)
soundfile.write(r"checks\check_1.wav", x[0].flatten(), SAMPLE_RATE)
