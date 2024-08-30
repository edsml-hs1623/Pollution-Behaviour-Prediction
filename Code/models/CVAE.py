import torch.nn as nn


class Conv3DAutoencoder(nn.Module):
    """
    A 3D Convolutional Autoencoder designed for input data with 3 channels and dimensions (64x64x64).
    This autoencoder compresses the input data into a lower-dimensional representation using a series of
    convolutional and pooling layers (encoder), and then reconstructs the data back to its original shape
    using transposed convolutional layers (decoder).

    Attributes:
    ----------
    encoder : nn.Sequential
        The encoding part of the autoencoder, which compresses the input data into a smaller representation.
        - Conv3d -> BatchNorm3d -> ReLU -> MaxPool3d layers are applied in sequence.
    
    decoder : nn.Sequential
        The decoding part of the autoencoder, which reconstructs the compressed data back to its original size.
        - ConvTranspose3d -> ReLU layers are applied in sequence, with a final Softmax activation layer.

    Methods:
    -------
    forward(x)
        Passes the input through the encoder to compress it, then through the decoder to reconstruct it.

    Example:
    -------
    model = Conv3DAutoencoder()
    input_data = torch.randn(1, 3, 64, 64, 64)  # Example input: batch of size 1, 3 channels, 64x64x64 volume
    output_data = model(input_data)
    print(output_data.shape)  # Expected output shape: (1, 3, 64, 64, 64)
    """

    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input size: (3, 64, 64, 64)
            nn.Conv3d(3, 16, kernel_size=3, stride=1, padding=1),  # Output: (16, 64, 64, 64)
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # Output: (16, 32, 32, 32)

            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),  # Output: (32, 32, 32, 32)
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # Output: (32, 16, 16, 16)

            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (64, 16, 16, 16)
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d((2, 1, 1)),  # Output: (64, 8, 8, 8)
        )

        # Decoder
        self.decoder = nn.Sequential(
            # Input: (64, 8, 8, 8)
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=(2, 1, 1), padding=1, output_padding=(1, 0, 0)),  # Output: (32, 16, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),  # Output: (16, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose3d(16, 3, kernel_size=3, stride=(2, 2, 2), padding=1, output_padding=(1, 1, 1)),  # Output: (3, 64, 64, 64)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Defines the forward pass through the autoencoder.
        
        Parameters:
        ----------
        x : torch.Tensor
            The input tensor of shape (batch_size, 3, 64, 64, 64).
        
        Returns:
        -------
        torch.Tensor
            The reconstructed output tensor of shape (batch_size, 3, 64, 64, 64).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
