import os
import pickle
import soundfile
import numpy as np
from vae import VAE

variational_autoencoder = VAE.load(save_folder=r"c:\Code\sound-of-cnns\model")
variational_autoencoder.summary()

DENORMALIZATION_PARAMETERS_PATH = r"C:\Code\sound-of-cnns\crafting_the_dataset\metadata\mean_and_stddev.pkl"
OUTPUT_PATH = "samples"

with open(DENORMALIZATION_PARAMETERS_PATH, "rb") as f:
    denormalization_parameters = pickle.load(f)


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

    signal = generated_data * stddev + mean
    return generated_data, signal


# 'generated_samples' contains the randomly generated data points in the original data space.
generated_samples, signal_samples = sample_from_latent_space_and_feed_to_decoder(vae_model=variational_autoencoder,
                                                                                 denormalize_params=denormalization_parameters,
                                                                                 num_of_samples=1
                                                                                 )
# TODO: FIX THE OUTPUT DIMENSION SO THAT IT WILL MATCH THE INPUT
# Reshape
reshaped = signal_samples[0].reshape((613888,))
# Save the generated data
soundfile.write(os.path.join(OUTPUT_PATH, "sample_1.wav"), reshaped, 22050)