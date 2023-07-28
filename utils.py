import os
import librosa
import numpy as np

SAMPLE_RATE = 44100
MONO = True


def read_audio_files(files_dir):
    metadata_cluster = {}
    for root, _, files in os.walk(files_dir):
        for file in files:
            file_path = os.path.join(root, file)

            file_duration = librosa.get_duration(path=file_path)
            signal, sample_rate = librosa.load(file_path, mono=MONO, sr=SAMPLE_RATE)
            rms = np.sqrt(np.mean(signal ** 2))

            num_channels = len(signal.shape)
            if num_channels == 1:
                channels = 1
            elif num_channels == 2:
                channels = 2
            else:
                print("unknown number of channels.")

            metadata_cluster[file] = {
                "path": file_path,
                "duration": file_duration,
                "samplerate": sample_rate,
                "rms": rms,
                "channels": channels
            }
            yield signal, metadata_cluster
