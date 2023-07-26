import numpy as np
from vae import VAE

variational_autoencoder = VAE.load(save_folder=r"c:\Code\sound-of-cnns\model")
variational_autoencoder.summary()


def sample_from_latent_space(vae_model, num_of_samples=1):
    # Get the dimensions of the latent space
    latent_dim = vae_model.latent_space_dim

    # Generate random values for the latent space (sample from Gaussian distribution)
    sampled_latent_vectors = np.random.normal(size=(num_of_samples, latent_dim))

    # Use the VAEs decoder to generate new data points from the sampled latent vectors
    generated_data = vae_model.decoder.predict(sampled_latent_vectors)

    return generated_data


# 'generated_samples' contains the randomly generated data points in the original data space.
generated_samples = sample_from_latent_space(vae_model=variational_autoencoder, num_of_samples=1)
