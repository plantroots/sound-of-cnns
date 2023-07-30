import os
import wave
import pickle
import soundfile
import numpy as np

from vae import VAE
from train import average_list_elements

variational_autoencoder = VAE.load(save_folder=r"c:\Code\sound-of-cnns\model")
variational_autoencoder.summary()

DENORMALIZATION_PARAMETERS_PATH = r"C:\Code\sound-of-cnns\crafting_the_dataset\metadata\train_set_mean_stddev_and_max_values.pkl"
OUTPUT_PATH = "samples"

with open(DENORMALIZATION_PARAMETERS_PATH, "rb") as f:
    denormalization_parameters = pickle.load(f)


def revert_peak_amplitude_normalization(normalized_data, original_max):
    # Get the maximum absolute amplitude from the normalized data
    max_val = np.max(np.abs(normalized_data))

    # Calculate the scaling factor used during normalization
    scaling_factor = original_max / max_val

    # Revert the normalization by multiplying with the scaling factor
    original_data = normalized_data * scaling_factor

    return original_data


def sample_from_latent_space_and_feed_to_decoder(vae_model, denormalize_params, num_of_samples=1):
    # Get the dimensions of the latent space
    latent_dim = vae_model.latent_space_dim

    # Generate random values for the latent space (sample from Gaussian distribution)
    sampled_latent_vectors = np.random.normal(size=(num_of_samples, latent_dim))

    # Use the VAEs decoder to generate new data points from the sampled latent vectors
    generated_data = vae_model.decoder.predict(sampled_latent_vectors)

    # Denormalize
    mean = denormalize_params["mean"]
    stddev = denormalize_params["stddev"]
    # max_values_mean = round(average_list_elements(denormalize_params["max_values"]), 2)
    max_values_mean = 0.91

    signal = revert_peak_amplitude_normalization(generated_data, max_values_mean)

    return generated_data, signal


SAMPLES_NUM = 10

# 'generated_samples' contains the randomly generated data points in the original data space.
generated_samples, signal_samples = sample_from_latent_space_and_feed_to_decoder(vae_model=variational_autoencoder,
                                                                                 denormalize_params=denormalization_parameters,
                                                                                 num_of_samples=SAMPLES_NUM
                                                                                 )


# def new_write_function(signal, index, directory):
#     num_channels = 1  # For mono audio, use 2 for stereo
#     sample_width = 4  # 2 bytes for 16-bit audio, 4 bytes for 32-bit audio
#     frame_rate = 22050  # Sample rate (samples per second)
#     num_samples = len(signal)  # Replace 'audio_data' with your NumPy array
#     print(num_samples)
#     with wave.open(os.path.join(directory, f"sample_{index}.wav"), "wb") as wav_file:
#         wav_file.setnchannels(num_channels)
#         wav_file.setsampwidth(sample_width)
#         wav_file.setframerate(frame_rate)
#         wav_file.setnframes(num_samples)
#         wav_file.writeframes(signal.tobytes())


for i in range(SAMPLES_NUM):
    # Reshape
    reshaped_signal = signal_samples[i].flatten()
    reshaped_generated = generated_samples[i].flatten()
    # print(len(reshaped))
    # Save the generated data

    # new_write_function(reshaped_signal, i, OUTPUT_PATH)
    soundfile.write(os.path.join(OUTPUT_PATH, f"demormalized_sample_{i}.wav"), reshaped_signal, 22050, format='wav')

    # soundfile.write(os.path.join(OUTPUT_PATH, f"generated_sample_{i}.wav"), reshaped_generated, 22050)
