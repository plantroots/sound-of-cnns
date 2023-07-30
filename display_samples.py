import librosa
import matplotlib.pyplot as plt

from utils import read_audio_files
from train import peak_amplitude_normalization

SAMPLE_RATE = 22050
# SAMPLE_RATE = 44100

# generated samples
AUDIO_FILES_DIR = r"c:\Code\sound-of-cnns\samples"

# original samples
# AUDIO_FILES_DIR = r"c:\Dataset\filtered_kicks_small"


def display_waveforms(audio_dir, normalization=False):
    plt.figure(figsize=(10, 4))
    for signal, metadata in read_audio_files(audio_dir):
        if normalization:
            signal = peak_amplitude_normalization(signal)
        librosa.display.waveshow(signal, sr=SAMPLE_RATE)
        plt.title('Waveform')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        plt.show()


display_waveforms(AUDIO_FILES_DIR, normalization=False)
