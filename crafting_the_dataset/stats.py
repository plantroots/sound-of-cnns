"""Check the sample rate, mono/stereo status, duration and amplitude"""

import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

AUDIO_FILES_DIR = r"c:\Dataset\kicks"
SHOW_PLOT = False
FFT_LIB = librosa.get_fftlib()

"""
Outputs and object that looks like this:

    'kick_2826.wav': {'path': 'c:\\Dataset\\kicks\\kick_2826.wav', 
                      'duration': 0.8173333333333334, 
                      'samplerate': 48000}
"""

file_cluster = {}
for root, _, files in os.walk(AUDIO_FILES_DIR):
    for file in files:
        file_path = os.path.join(root, file)
        file_duration = librosa.get_duration(path=file_path)
        file_samplerate = librosa.get_samplerate(path=file_path)
        file_cluster[file] = {"path": file_path,
                              "duration": file_duration,
                              "samplerate": file_samplerate}

files = tuple(file_cluster.values())
stats_df = pd.DataFrame.from_records(files)

unique_sample_rates = stats_df["samplerate"].unique().tolist()
num_of_unique_sample_rates = len(unique_sample_rates)


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


print("\n", f" -> stats for: {AUDIO_FILES_DIR} <-", "\n")
print("*" * 25)
statistic_print("duration")
statistic_print("samplerate", show_unique=True)
print("*" * 25)

if SHOW_PLOT:
    plt.hist(stats_df['samplerate'], bins=num_of_unique_sample_rates)
    column_name = stats_df['samplerate'].name
    plt.title(f'Histogram of {column_name}')
    plt.show()
