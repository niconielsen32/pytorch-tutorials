#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Generative Models in PyTorch

This script demonstrates implementations of various generative models using PyTorch,
including Autoencoders (AEs), Variational Autoencoders (VAEs), and Generative
Adversarial Networks (GANs). It also provides a conceptual overview of Diffusion Models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directory
output_dir = "09_generative_models_outputs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "generated_images"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)

# MNIST Dataset (common for generative model demos)
img_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)) # Normalize to [-1, 1] if using Tanh in generator output
    # For Sigmoid output in generator, [0,1] from ToTensor() is fine.
])

mnist_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=img_transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=128, shuffle=True, num_workers=2 if os.name == 'posix' else 0)

# Image dimensions for MNIST
IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = 28, 28, 1
IMG_SHAPE = (IMG_CHANNELS, IMG_WIDTH, IMG_HEIGHT)
FLATTENED_IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * IMG_CHANNELS

# ---------------------------------------------------------------------------
# Section 1: Introduction to Generative Models (Conceptual - in README)
# ---------------------------------------------------------------------------
def intro_generative_models():
    print("\nSection 1: Introduction to Generative Models")
    print("-" * 70)
    print("This section is conceptual and detailed in the README.md.")
    print("Covers: What are Generative Models, Importance, Taxonomy.")

# ---------------------------------------------------------------------------
# Section 2: Autoencoders (AEs)
# ---------------------------------------------------------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim=FLATTENED_IMG_SIZE, latent_dim=32):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),  # Latent space representation
            nn.ReLU(True)  # Or Tanh, depends on desired latent space properties
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # To output values between 0 and 1 for image pixels
        )

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)  # Flatten input image
        encoded = self.encoder(x_flat)
        decoded_flat = self.decoder(encoded)
        decoded = decoded_flat.view(x.size(0), *IMG_SHAPE)  # Reshape to original image shape
        return decoded, encoded  # Return reconstructed image and latent code


def train_autoencoder(model, dataloader, num_epochs=5, lr=1e-3):
    print("Training Autoencoder...")
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (imgs, _) in enumerate(dataloader):  # Labels are not used for AE
            imgs = imgs.to(device)
            optimizer.zero_grad()
            reconstructed_imgs, _ = model(imgs)
            loss = criterion(reconstructed_imgs, imgs)  # Compare reconstruction with original
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_epoch_loss)
        print(f"=> AE Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
    print(f"Autoencoder training finished in {time.time()-start_time:.2f}s")
    return losses


def demonstrate_autoencoder():
    print("\nSection 2: Autoencoders (AEs)")
    print("-" * 70)
    ae_model = Autoencoder(latent_dim=64).to(device)
    print("Autoencoder Architecture:")
    print(ae_model)
    
    # Train the Autoencoder (short training for demo)
    ae_losses = train_autoencoder(ae_model, mnist_loader, num_epochs=3)  # Increase epochs for better results
    if ae_losses:
        plt.figure(figsize=(8, 4))
        plt.plot(ae_losses, label="AE Training Loss (MSE)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Autoencoder Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "plots", "autoencoder_loss.png"))
        plt.close()
        print("AE training loss plot saved.")

    # Visualize some reconstructions
    ae_model.eval()
    with torch.no_grad():
        test_imgs, _ = next(iter(mnist_loader))
        test_imgs = test_imgs.to(device)[:16]  # Take 16 images for visualization
        reconstructed_test_imgs, _ = ae_model(test_imgs)
        comparison = torch.cat([test_imgs.view(-1, 1, IMG_WIDTH, IMG_HEIGHT),
                                reconstructed_test_imgs.view(-1, 1, IMG_WIDTH, IMG_HEIGHT)])
        save_image(comparison.cpu(), os.path.join(output_dir, "generated_images", "ae_reconstruction.png"), nrow=16)
        print("AE original vs reconstructed images saved.")

# ---------------------------------------------------------------------------
# Section 3: Variational Autoencoders (VAEs)
# ---------------------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, input_dim=FLATTENED_IMG_SIZE, h_dim=400, z_dim=20):  # z_dim is latent space dimension
        super(VAE, self).__init__()
        self.fc_enc1 = nn.Linear(input_dim, h_dim)
        self.fc_enc_mean = nn.Linear(h_dim, z_dim)  # To output mu (mean)
        self.fc_enc_logvar = nn.Linear(h_dim, z_dim)  # To output log_var (log variance)
        
        self.fc_dec1 = nn.Linear(z_dim, h_dim)
        self.fc_dec2 = nn.Linear(h_dim, input_dim)  # To reconstruct image

    def encode(self, x):
        h = F.relu(self.fc_enc1(x))
        return self.fc_enc_mean(h), self.fc_enc_logvar(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # std = sqrt(variance) = exp(0.5 * log_var)
        eps = torch.randn_like(std)  # Sample epsilon from N(0, I)
        return mu + eps * std  # Sample z from N(mu, var)

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        return torch.sigmoid(self.fc_dec2(h))  # Use sigmoid for pixel values [0,1]

    def forward(self, x):
        x_flat = x.view(x.size(0), -1)
        mu, log_var = self.encode(x_flat)
        z = self.reparameterize(mu, log_var)
        x_reconstructed_flat = self.decode(z)
        x_reconstructed = x_reconstructed_flat.view(x.size(0), *IMG_SHAPE)
        return x_reconstructed, mu, log_var


# VAE Loss = Reconstruction Loss + KL Divergence
def vae_loss_function(recon_x, x, mu, log_var):
    # Reconstruction loss (e.g., Binary Cross Entropy for sigmoid output)
    BCE = F.binary_cross_entropy(recon_x.view(-1, FLATTENED_IMG_SIZE),
                                 x.view(-1, FLATTENED_IMG_SIZE), reduction='sum')
    # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def train_vae(model, dataloader, num_epochs=5, lr=1e-3):
    print("Training VAE...")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon_imgs, mu, log_var = model(imgs)
            loss = vae_loss_function(recon_imgs, imgs, mu, log_var)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if batch_idx % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item()/imgs.size(0):.4f}")
        avg_epoch_loss = epoch_loss / len(dataloader.dataset)
        losses.append(avg_epoch_loss)
        print(f"=> VAE Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.4f}")
    print(f"VAE training finished in {time.time()-start_time:.2f}s")
    return losses


def demonstrate_vae():
    print("\nSection 3: Variational Autoencoders (VAEs)")
    print("-" * 70)
    vae_model = VAE(z_dim=20).to(device)  # Latent dimension z_dim=20
    print("VAE Architecture:")
    print(vae_model)

    vae_losses = train_vae(vae_model, mnist_loader, num_epochs=3)  # Increase epochs for better results
    if vae_losses:
        plt.figure(figsize=(8, 4))
        plt.plot(vae_losses, label="VAE Training Loss (Reconstruction + KLD)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VAE Training Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "plots", "vae_loss.png"))
        plt.close()
        print("VAE training loss plot saved.")

    # Visualize reconstructions and generated samples
    vae_model.eval()
    with torch.no_grad():
        # Reconstructions
        test_imgs_vae, _ = next(iter(mnist_loader))
        test_imgs_vae = test_imgs_vae.to(device)[:8]  # 8 images for reconstruction viz
        reconstructed_vae, _, _ = vae_model(test_imgs_vae)
        comparison_vae = torch.cat([test_imgs_vae.view(-1, 1, IMG_WIDTH, IMG_HEIGHT),
                                    reconstructed_vae.view(-1, 1, IMG_WIDTH, IMG_HEIGHT)])
        save_image(comparison_vae.cpu(), os.path.join(output_dir, "generated_images", "vae_reconstruction.png"), nrow=8)
        print("VAE original vs reconstructed images saved.")

        # Generate new samples from latent space
        num_generated_samples = 16
        random_latent_z = torch.randn(num_generated_samples, 20).to(device)  # Sample z from N(0,I)
        generated_imgs_flat = vae_model.decode(random_latent_z)
        generated_imgs = generated_imgs_flat.view(num_generated_samples, *IMG_SHAPE)
        save_image(generated_imgs.cpu(), os.path.join(output_dir, "generated_images", "vae_generated_samples.png"), nrow=4)
        print("VAE generated samples from random latent vectors saved.")

# ---------------------------------------------------------------------------
# Section 4: Generative Adversarial Networks (GANs)
# ---------------------------------------------------------------------------

# --- Generator for GAN ---
class GANGenerator(nn.Module):
    def __init__(self, z_dim=100, img_dim=FLATTENED_IMG_SIZE):  # z_dim is latent noise vector size
        super(GANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_dim),
            nn.Tanh()  # Output pixel values in [-1, 1] - requires input data normalization to [-1,1]
            # If using Sigmoid, output is [0,1] and input data should be [0,1]
        )

    def forward(self, z):  # z is noise vector
        img_flat = self.model(z)
        img = img_flat.view(img_flat.size(0), *IMG_SHAPE)  # Reshape to image
        return img


# --- Discriminator for GAN ---
class GANDiscriminator(nn.Module):
    def __init__(self, img_dim=FLATTENED_IMG_SIZE):
        super(GANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability: Real (1) or Fake (0)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)  # Flatten image
        validity = self.model(img_flat)
        return validity


def train_gan(generator, discriminator, dataloader, num_epochs=10, lr=0.0002, z_dim=100):
    print("Training GAN...")
    # Loss function: Binary Cross Entropy
    adversarial_loss = nn.BCELoss()
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    g_losses, d_losses = [], []
    fixed_noise = torch.randn(64, z_dim).to(device)  # For consistent visualization of G's progress
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_g_loss, epoch_d_loss = 0.0, 0.0
        for i, (real_imgs, _) in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            # For Tanh output, normalize real_imgs to [-1, 1]
            # real_imgs = (real_imgs - 0.5) * 2.0
            # If Generator uses Sigmoid, real_imgs are fine as [0,1]
            batch_size = real_imgs.size(0)
            # Adversarial ground truths
            real_label = torch.full((batch_size, 1), 1.0, device=device, dtype=torch.float)
            fake_label = torch.full((batch_size, 1), 0.0, device=device, dtype=torch.float)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            # Loss for real images
            real_pred = discriminator(real_imgs)
            d_real_loss = adversarial_loss(real_pred, real_label)
            # Loss for fake images
            z_noise = torch.randn(batch_size, z_dim).to(device)
            fake_imgs = generator(z_noise)
            fake_pred = discriminator(fake_imgs.detach())  # detach to avoid training G on D's loss
            d_fake_loss = adversarial_loss(fake_pred, fake_label)
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            optimizer_G.zero_grad()
            # Generate fake images again (new batch, or use the one above if not detached)
            # z_noise = torch.randn(batch_size, z_dim).to(device) # Can reuse or make new
            # fake_imgs = generator(z_noise) # No need to generate again if we used .detach() correctly
            fake_pred_for_g = discriminator(fake_imgs)  # Pass fake_imgs (not detached) through D
            g_loss = adversarial_loss(fake_pred_for_g, real_label)  # Generator wants D to think fake is real
            g_loss.backward()
            optimizer_G.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()

            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs} | Batch {i+1}/{len(dataloader)} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

        avg_epoch_d_loss = epoch_d_loss / len(dataloader)
        avg_epoch_g_loss = epoch_g_loss / len(dataloader)
        d_losses.append(avg_epoch_d_loss)
        g_losses.append(avg_epoch_g_loss)
        print(f"=> GAN Epoch {epoch+1}/{num_epochs}, Avg D Loss: {avg_epoch_d_loss:.4f}, Avg G Loss: {avg_epoch_g_loss:.4f}")
        
        # Save generated image samples from fixed noise
        if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:  # Save every 2 epochs and last epoch
            with torch.no_grad():
                generator.eval()
                generated_fixed_noise = generator(fixed_noise).detach().cpu()
                # If generator used Tanh, unnormalize from [-1,1] to [0,1] for saving
                # generated_fixed_noise = generated_fixed_noise * 0.5 + 0.5
                save_image(generated_fixed_noise,
                           os.path.join(output_dir, "generated_images", f"gan_epoch_{epoch+1}.png"),
                           nrow=8, normalize=True if generator.model[-1].__class__.__name__ == "Tanh" else False)
                generator.train()
    print(f"GAN training finished in {time.time()-start_time:.2f}s")
    return g_losses, d_losses


def demonstrate_gan():
    print("\nSection 4: Generative Adversarial Networks (GANs)")
    print("-" * 70)
    Z_DIMENSION = 100  # Latent dimension for noise vector

    gan_generator = GANGenerator(z_dim=Z_DIMENSION).to(device)
    gan_discriminator = GANDiscriminator().to(device)
    print("GAN Generator Architecture:")
    print(gan_generator)
    print("GAN Discriminator Architecture:")
    print(gan_discriminator)

    # Train GAN (short training for demo)
    # Note: GANs often require many epochs (50-200+) and careful hyperparameter tuning.
    # Using normalized MNIST data if Tanh is in generator output
    # For this demo, using Sigmoid in generator and ToTensor() is fine for MNIST [0,1]
    # If Tanh, then need: transforms.Normalize((0.5,), (0.5,))
    # And unnormalize when saving: img = img * 0.5 + 0.5
    g_losses, d_losses = train_gan(gan_generator, gan_discriminator, mnist_loader, num_epochs=5, z_dim=Z_DIMENSION)

    if g_losses and d_losses:
        plt.figure(figsize=(10, 5))
        plt.plot(g_losses, label="Generator Loss")
        plt.plot(d_losses, label="Discriminator Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (BCE)")
        plt.title("GAN Training Losses")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "plots", "gan_training_losses.png"))
        plt.close()
        print("GAN training losses plot saved.")

    print("Generated image samples from GAN training are saved in the outputs directory.")

# ---------------------------------------------------------------------------
# Section 5: Introduction to Diffusion Models (Conceptual - in README)
# ---------------------------------------------------------------------------
def intro_diffusion_models():
    print("\nSection 5: Introduction to Diffusion Models")
    print("-" * 70)
    print("This section is conceptual and detailed in the README.md.")
    print("Covers: Forward (Noising) Process, Reverse (Denoising) Process, U-Net architecture.")
    print("Diffusion models are known for high-quality sample generation but can be complex to implement fully.")

# ---------------------------------------------------------------------------
# Section 6: Applications of Generative Models (Conceptual - in README)
# ---------------------------------------------------------------------------
def applications_generative_models():
    print("\nSection 6: Applications of Generative Models")
    print("-" * 70)
    print("This section is conceptual and detailed in the README.md.")
    print("Covers: Image Synthesis/Editing, Text Generation, Drug Discovery, Anomaly Detection, Data Augmentation.")

# ---------------------------------------------------------------------------
# Main function to run selected demonstrations
# ---------------------------------------------------------------------------
def main():
    """Main function to run Generative Models tutorial sections."""
    print("=" * 80)
    print("PyTorch Generative Models Tutorial")
    print("=" * 80)

    intro_generative_models()  # Section 1
    demonstrate_autoencoder()   # Section 2
    demonstrate_vae()           # Section 3
    demonstrate_gan()           # Section 4
    intro_diffusion_models()    # Section 5 (Conceptual)
    applications_generative_models()  # Section 6 (Conceptual)

    print("\nTutorial complete! Outputs (generated images, plots) are in '09_generative_models_outputs' directory.")

if __name__ == '__main__':
    main()
