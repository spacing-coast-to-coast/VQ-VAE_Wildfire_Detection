import torch
import torch.nn as nn

# Encoder: Converts the input time series window to a latent representation.
class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        z_e = self.conv2(x)
        return z_e

# Vector Quantizer: Maps continuous latent vectors to discrete codebook embeddings.
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z_e):
        # Rearrange z_e from (batch, latent_dim, length) to (batch*length, latent_dim)
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)
        # Compute distances between z_e vectors and embedding vectors
        distances = (z_e_flat.pow(2).sum(1, keepdim=True)
                     + self.embedding.weight.pow(2).sum(1)
                     - 2 * torch.matmul(z_e_flat, self.embedding.weight.t()))
        # Find the closest embedding index for each latent vector
        encoding_indices = torch.argmin(distances, dim=1)
        # Quantize the latent vectors using the embedding table
        z_q = self.embedding(encoding_indices).view(z_e.shape)
        
        # Calculate the loss terms
        loss = self.commitment_cost * torch.mean((z_e.detach() - z_q)**2) + \
               torch.mean((z_e - z_q.detach())**2)
        # Use the straight-through estimator for the backward pass
        z_q = z_e + (z_q - z_e).detach()
        return z_q, loss

# Decoder: Reconstructs the time series window from the quantized latent codes.
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, output_channels, kernel_size=3, padding=1)
        
    def forward(self, z_q):
        x = self.relu(self.conv1(z_q))
        x_recon = self.conv2(x)
        return x_recon

# VQ-VAE: Combines the Encoder, Vector Quantizer, and Decoder.
class VQVAE(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, input_channels)
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

