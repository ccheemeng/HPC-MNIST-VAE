import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, z_dim):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(x_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.log_sigma = nn.Linear(hidden_dim, z_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        h = self.leaky_relu(self.linear1(x))
        mean = self.mu(h)
        log_var = self.log_sigma(h)
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, x_dim)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, z):
        h = self.leaky_relu(self.linear1(z))
        x_hat = torch.sigmoid(self.output(h))
        return x_hat

class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterise(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterise(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var