import os
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
    max_values_mean = round(average_list_elements(denormalize_params["max_values"]), 2)

    signal = revert_peak_amplitude_normalization(generated_data, max_values_mean)

    return generated_data, signal


# 'generated_samples' contains the randomly generated data points in the original data space.
generated_samples, signal_samples = sample_from_latent_space_and_feed_to_decoder(vae_model=variational_autoencoder,
                                                                                 denormalize_params=denormalization_parameters,
                                                                                 num_of_samples=1
                                                                                 )
# Reshape
reshaped_signal = signal_samples[0].flatten()
reshaped_generated = generated_samples[0].flatten()
# print(len(reshaped))
# Save the generated data
soundfile.write(os.path.join(OUTPUT_PATH, "demormalized_sample_1.wav"), reshaped_signal, 22050)
soundfile.write(os.path.join(OUTPUT_PATH, "generated_sample_1.wav"), reshaped_generated, 22050)

