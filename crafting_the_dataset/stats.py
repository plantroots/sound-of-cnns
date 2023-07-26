"""Check the duration, sample rate, mono/stereo status and rms/loudness"""

import os
import shutil
import pickle
import random
import librosa

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AUDIO_FILES_DIR = r"c:\Dataset\kicks"
FILTERED_FILES_DIR = r"c:\Dataset\filtered_kicks\sounds"

SAMPLE_RATE = 44100
MONO = True
DURATION_THRESHOLD_IN_SECONDS = 1.74

SHOW_PLOT = True
FFT_LIB = librosa.get_fftlib()


def read_files_metadata(files_dir, save_to_disk=False):
    """
    Outputs and object that looks like this:

        'kick_2826.wav': {'path': 'c:\\Dataset\\kicks\\kick_2826.wav',
                          'duration': 0.8173333333333334,
                          'samplerate': 48000}
    """
    metadata_cluster = {}
    for root, _, files in os.walk(files_dir):
        for file in files:
            file_path = os.path.join(root, file)

            # DURATION
            file_duration = librosa.get_duration(path=file_path)

            # MONO/STEREO status
            channels = None

            # MAKE all MONO and the same SAMPLERATE
            # audio_data, sample_rate = librosa.load(file_path)
            audio_data, sample_rate = librosa.load(file_path, mono=MONO, sr=SAMPLE_RATE)

            num_channels = len(audio_data.shape)
            if num_channels == 1:
                channels = 1
            elif num_channels == 2:
                channels = 2
            else:
                print("unknown number of channels.")

            # RMS/LOUDNESS
            rms = np.sqrt(np.mean(audio_data ** 2))

            metadata_cluster[file] = {
                "path": file_path,
                "duration": file_duration,
                "samplerate": sample_rate,
                "channels": channels,
                "rms": rms
            }

    if save_to_disk:
        os.makedirs("metadata") if not os.path.exists("metadata") else None
        with open(r"metadata\audio_metadata.pkl", "wb") as f:
            pickle.dump(metadata_cluster, f)
    return metadata_cluster


def pick_a_random_waveform_and_display_it(signals):
    random_index = random.randint(0, len(signals) - 1)
    randomly_picked_array = signals[random_index]
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(randomly_picked_array, sr=SAMPLE_RATE)
    plt.title('Waveform')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()


def statistic_print(dataframe_column_name, show_unique=False):
    print(f"{dataframe_column_name.upper()}:")
    print("-" * 10)
    print("mean:", round(stats_df[dataframe_column_name].mean(), 3))
    print("stddev:", round(stats_df[dataframe_column_name].std(), 3))
    print("min:", round(stats_df[dataframe_column_name].min(), 3))
    print("max:", round(stats_df[dataframe_column_name].max(), 3))
    if show_unique:
        print("unique value counts:", stats_df[dataframe_column_name].value_counts().to_dict())
    print("-" * 10, "\n")


def histogram(dataframe_column):
    column_name = dataframe_column.name
    unique = len(dataframe_column.unique().tolist())
    plt.style.use('grayscale')
    plt.hist(dataframe_column, bins=unique)
    plt.xlabel(column_name)
    plt.ylabel('count')
    plt.title(column_name)
    plt.show()


def filter_based_on_duration_in_seconds(metadata_object, duration_threshold):
    filtered = []
    filtered_full_paths = []
    for file_name, metadata in metadata_object.items():
        if metadata["duration"] <= duration_threshold:
            signal, _ = librosa.load(metadata["path"], mono=MONO, sr=SAMPLE_RATE)
            filtered.append(signal)
            filtered_full_paths.append(metadata["path"])
    return filtered, len(filtered), filtered_full_paths


def add_padding(signals):
    # TODO: adding padding to the right only
    max_length = DURATION_THRESHOLD_IN_SECONDS
    signals_padded = []
    for sig in signals:
        signal_padded = librosa.util.pad_center(sig, size=int(max_length * SAMPLE_RATE))
        signals_padded.append(signal_padded)
    return signals_padded


def copy_file(source_path, destination_path):
    try:
        # Check if the source file exists
        if not os.path.isfile(source_path):
            print(f"Error: Source file '{source_path}' does not exist.")
            return

        # Perform the file copy operation
        shutil.copy(source_path, destination_path)
        print(f"File '{source_path}' copied to '{destination_path}' successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")


def copy_files_to_destination_folder(file_paths, destination_folder):
    for path in file_paths:
        file_name = os.path.basename(path)
        destination_path = os.path.join(destination_folder, file_name)
        copy_file(path, destination_path)


# LOADING/READING the audio data
with open(r"metadata\audio_metadata.pkl", 'rb') as f:
    file_cluster = pickle.load(f)
# file_cluster = read_files_metadata(AUDIO_FILES_DIR, save_to_disk=True)

# FILTER duration (2826 in total)
# 1.15 with 1x stddev and 1.74 with 2x stddev
# at 1.74 -> 76734 samples at this samplerate
signals_filtered, num_of_signals_after_filtering, filtered_paths = filter_based_on_duration_in_seconds(file_cluster,
                                                                                                       DURATION_THRESHOLD_IN_SECONDS)
print("# of signals before filtering:", len(file_cluster.keys()))
print("# of signals after filtering:", num_of_signals_after_filtering)
print("# of signals removed:", len(file_cluster.keys()) - num_of_signals_after_filtering)

# SAVE TO DISK FILTERED
copy_files_to_destination_folder(filtered_paths, FILTERED_FILES_DIR)

# PADDING
signals_filtered_padded = add_padding(signals_filtered)

# STATS DATAFRAME
file_cluster_tuple = tuple(file_cluster.values())
stats_df = pd.DataFrame.from_records(file_cluster_tuple)

# DISPLAY random waveform
# pick_a_random_waveform_and_display_it(signals_filtered_padded)

print("\n", f" -> stats for: {AUDIO_FILES_DIR} <-", "\n")
print("*" * 25)
# statistic_print("duration")
# statistic_print("samplerate", show_unique=True)
# statistic_print("rms")
# statistic_print("channels", show_unique=True)
print("*" * 25)

# if SHOW_PLOT:
#     histogram(stats_df["samplerate"])
#     histogram(stats_df["duration"])
#     histogram(stats_df["rms"])
