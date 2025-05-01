import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

class SparseAutoencoder(nn.Module):
    """
    Simple Sparse Autoencoder (SAE) with ReLU activation and optional tied weights.
    """
    def __init__(self, cfg: DictConfig):
        """
        Args:
            cfg (DictConfig): Configuration object with model parameters:
                - input_dim (int): Dimension of the input features.
                - hidden_dim_factor (int): Factor to determine hidden dim (hidden = input * factor).
                - tied_weights (bool): Whether to tie encoder and decoder weights.
                - activation_fn (str): Activation function ('relu').
                - encoder_bias (bool): Whether to use bias in the encoder.
                - decoder_bias (bool): Whether to use bias in the decoder.
        """
        super().__init__()
        self.input_dim = cfg.input_dim
        self.hidden_dim = self.input_dim * cfg.hidden_dim_factor
        self.tied_weights = cfg.tied_weights
        self.activation_fn_name = cfg.activation_fn
        self.use_encoder_bias = cfg.encoder_bias
        self.use_decoder_bias = cfg.decoder_bias

        # Encoder weights and bias
        self.W_e = nn.Parameter(torch.Tensor(self.hidden_dim, self.input_dim))
        if self.use_encoder_bias:
            self.b_e = nn.Parameter(torch.Tensor(self.hidden_dim))
        else:
            self.register_parameter('b_e', None)

        # Decoder weights and bias
        if not self.tied_weights:
            self.W_d = nn.Parameter(torch.Tensor(self.input_dim, self.hidden_dim))
        # If tied, W_d is implicitly W_e.T

        if self.use_decoder_bias:
            self.b_d = nn.Parameter(torch.Tensor(self.input_dim))
        else:
            self.register_parameter('b_d', None)

        self.init_weights()
        self._set_activation_fn()

    def init_weights(self):
        # Initialize weights using Kaiming He initialization (good for ReLU)
        nn.init.kaiming_uniform_(self.W_e, a=0, mode='fan_in', nonlinearity='relu') # Assuming ReLU
        if self.b_e is not None:
            nn.init.zeros_(self.b_e)

        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.W_d, a=0, mode='fan_in', nonlinearity='relu') # Assuming ReLU activation before decoder

        if self.b_d is not None:
            nn.init.zeros_(self.b_d)

    def _set_activation_fn(self):
        if self.activation_fn_name.lower() == 'relu':
            self.activation_fn = F.relu
        # Add other activations here if needed ('gelu', etc.)
        # elif self.activation_fn_name.lower() == 'gelu':
        #     self.activation_fn = F.gelu
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn_name}")

    def encode(self, x):
        """Encodes the input tensor."""
        # Ensure x is float32 for matmul
        x = x.to(self.W_e.dtype)
        # Linear transformation
        encoded = F.linear(x, self.W_e, self.b_e)
        # Apply activation
        hidden_activations = self.activation_fn(encoded)
        return hidden_activations

    def decode(self, h):
        """Decodes the hidden activation tensor."""
        h = h.to(self.W_e.dtype if self.tied_weights else self.W_d.dtype)
        if self.tied_weights:
            # Use transpose of encoder weights
            decoded = F.linear(h, self.W_e.t(), self.b_d)
        else:
            decoded = F.linear(h, self.W_d, self.b_d)
        return decoded

    def forward(self, x):
        """
        Forward pass through the SAE.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            tuple: (reconstruction, hidden_activations)
                - reconstruction (torch.Tensor): Reconstructed input, shape (batch_size, input_dim).
                - hidden_activations (torch.Tensor): Activations of the hidden layer, shape (batch_size, hidden_dim).
        """
        hidden_activations = self.encode(x)
        reconstruction = self.decode(hidden_activations)
        return reconstruction, hidden_activations

    def calculate_loss(self, x, reconstruction, hidden_activations, lambda_sparsity):
        """
        Calculates the SAE loss.

        Args:
            x (torch.Tensor): Original input tensor.
            reconstruction (torch.Tensor): Reconstructed tensor.
            hidden_activations (torch.Tensor): Hidden layer activations.
            lambda_sparsity (float): Coefficient for the L1 sparsity penalty.

        Returns:
            tuple: (total_loss, reconstruction_loss, sparsity_loss)
        """
        # Ensure types match for loss calculation
        x = x.to(reconstruction.dtype)

        # Reconstruction Loss (MSE)
        reconstruction_loss = F.mse_loss(reconstruction, x, reduction='mean')

        # Sparsity Loss (L1 norm on hidden activations)
        sparsity_loss = torch.mean(torch.abs(hidden_activations)) # L1 norm averaged over batch and hidden dim

        # Total Loss
        total_loss = reconstruction_loss + lambda_sparsity * sparsity_loss

        return total_loss, reconstruction_loss, sparsity_loss

    @staticmethod
    @torch.no_grad()
    def calculate_l0_norm(hidden_activations, threshold=1e-6):
        """Estimates the L0 norm (average number of non-zero elements)."""
        return torch.mean((hidden_activations.abs() > threshold).float())
