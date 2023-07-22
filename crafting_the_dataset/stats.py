"""Check the sample rate, mono/stereo status, length and amplitude"""

import os
import librosa
import numpy as np
import pandas as pd

AUDIO_FILES_DIR = r"c:\Dataset\kicks"
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

