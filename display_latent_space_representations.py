import numpy as np
import matplotlib.pyplot as plt

from vae import VAE
from utils import read_audio_files
from train import peak_amplitude_normalization, NUM_OF_SAMPLES_IN_A_FILE

MODEL_PATH = r"c:\Models\1024_FS_5LS"
AUDIO_FILES_PATH = r"c:\Dataset\filtered_kicks_small"

vae = VAE.load(MODEL_PATH)

for signal, metadata in read_audio_files(AUDIO_FILES_PATH):

    normalized_signal = peak_amplitude_normalization(signal)

    if len(normalized_signal) < NUM_OF_SAMPLES_IN_A_FILE:
        padding_to_add = NUM_OF_SAMPLES_IN_A_FILE - len(normalized_signal)
        normalized_signal = np.append(normalized_signal, np.zeros(padding_to_add))

    normalized_signal = normalized_signal[np.newaxis, ...]
    normalized_signal = normalized_signal.reshape(1, 40960, 1)

    latent_representation = vae.encoder.predict(normalized_signal)
