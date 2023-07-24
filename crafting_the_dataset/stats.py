"""Check the duration, sample rate, mono/stereo status and rms/loudness"""

import os
import pickle
import random
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AUDIO_FILES_DIR = r"c:\Dataset\kicks"
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
            audio_data, sample_rate = librosa.load(file_path, mono=True, sr=44100)

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
    librosa.display.waveshow(randomly_picked_array, sr=44100)
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


# LOADING/READING the audio data
with open(r"metadata\audio_metadata.pkl", 'rb') as f:
    file_cluster = pickle.load(f)
# file_cluster = read_files_metadata(AUDIO_FILES_DIR, save_to_disk=True)

# FILTER SOME OF THE SIGNALS OUT (longer than 2 seconds) - 2826 total samples in the dataset
# 1.15 with 1x stddev and 1.74 with 2x stddev
# at 1.74 -> 76734 samples at this samplerate
DURATION_THRESHOLD_IN_SECONDS = 1.74
# max_length = round(max(v["duration"] for k, v in file_cluster.items()), 2)

signals_filtered = []
for file_name, metadata in file_cluster.items():
    if metadata["duration"] <= DURATION_THRESHOLD_IN_SECONDS:
        signal, _ = librosa.load(metadata["path"], mono=True, sr=44100)
        signals_filtered.append(signal)
print(len(signals_filtered))

# TODO: adding padding to the right only
# PAD TO THE SAME LENGTH THE FILTERED SIGNALS
# max_length = max(librosa.get_duration(y=sig, sr=44100) for sig in signals_filtered)
max_length = DURATION_THRESHOLD_IN_SECONDS
signals_filtered_padded = []
for sig in signals_filtered:
    signal_padded = librosa.util.pad_center(sig, size=int(max_length * 44100))
    signals_filtered_padded.append(signal_padded)

# check padded
ds = []
for s in signals_filtered_padded:
    ds.append(librosa.get_duration(y=s, sr=44100))
print(min(ds))
print(max(ds))

file_cluster_tuple = tuple(file_cluster.values())
stats_df = pd.DataFrame.from_records(file_cluster_tuple)

unique_sample_rates = len(stats_df["samplerate"].unique().tolist())

pick_a_random_waveform_and_display_it(signals_filtered_padded)

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
