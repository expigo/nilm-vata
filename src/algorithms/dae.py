"""
Denoising Autoencoder (DAE) for NILM

Uses autoencoders to learn appliance signatures and denoise aggregate signals.
"""
import numpy as np
import pandas as pd
from typing import Dict
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn(
        "PyTorch not available. Install with: pip install torch\n"
        "DAE algorithm will not work without PyTorch."
    )
    # Create dummy classes to avoid import errors
    class nn:
        class Module:
            pass


if TORCH_AVAILABLE:
    class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder network"""

    def __init__(self, window_size=100, encoding_dim=64):
        """
        Initialize DAE

        Args:
            window_size: Size of input window
            encoding_dim: Dimension of encoded representation
        """
        super(DenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, window_size),
            nn.ReLU()  # Ensure non-negative output
        )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input [batch, window_size]

        Returns:
            Reconstructed output [batch, window_size]
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DAEDisaggregator:
    """
    Denoising Autoencoder disaggregator

    Trains separate autoencoders for each appliance. Each autoencoder learns
    to extract that appliance's signal from the aggregate consumption.
    """

    def __init__(
        self,
        window_size: int = 100,
        encoding_dim: int = 64,
        learning_rate: float = 0.001,
        epochs: int = 20,
        batch_size: int = 64,
        noise_factor: float = 0.1
    ):
        """
        Initialize DAE disaggregator

        Args:
            window_size: Size of sliding window
            encoding_dim: Encoding dimension
            learning_rate: Learning rate
            epochs: Number of training epochs
            batch_size: Batch size
            noise_factor: Amount of noise to add during training
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DAE. Install with: pip install torch"
            )

        self.window_size = window_size
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_factor = noise_factor

        self.models = {}
        self.appliance_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Train DAE models for each appliance

        Args:
            mains: Aggregate power consumption
            appliances: Dictionary of appliance-wise power data
        """
        print(f"Training Denoising Autoencoder (device={self.device})...")
        print(f"  Window size: {self.window_size}")
        print(f"  Encoding dim: {self.encoding_dim}")
        print(f"  Epochs: {self.epochs}")

        self.appliance_names = list(appliances.keys())

        # Create windows from aggregate
        X_windows = self._create_windows(mains['power'].values)

        for app_name, app_data in appliances.items():
            print(f"\n  Training {app_name}...")

            # Create windows from appliance
            y_windows = self._create_windows(app_data['power'].values)

            # Align lengths
            min_len = min(len(X_windows), len(y_windows))
            X = X_windows[:min_len]
            y = y_windows[:min_len]

            # Normalize
            X_max = X.max()
            y_max = y.max() if y.max() > 0 else 1.0

            X_norm = X / X_max if X_max > 0 else X
            y_norm = y / y_max if y_max > 0 else y

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_norm).to(self.device)
            y_tensor = torch.FloatTensor(y_norm).to(self.device)

            # Create dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )

            # Initialize model
            model = DenoisingAutoencoder(
                window_size=self.window_size,
                encoding_dim=self.encoding_dim
            ).to(self.device)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            # Training loop
            model.train()
            for epoch in range(self.epochs):
                total_loss = 0

                for batch_X, batch_y in dataloader:
                    # Add noise to input (denoising)
                    noise = torch.randn_like(batch_X) * self.noise_factor
                    noisy_X = batch_X + noise
                    noisy_X = torch.clamp(noisy_X, 0, 1)

                    # Forward pass
                    outputs = model(noisy_X)
                    loss = criterion(outputs, batch_y)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)

                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

            # Store model with normalization factors
            self.models[app_name] = {
                'model': model,
                'X_max': X_max,
                'y_max': y_max
            }

            print(f"    Training complete!")

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Disaggregate using trained DAE models

        Args:
            mains: Aggregate power consumption
            num_samples: Number of samples to process

        Returns:
            Dictionary of predicted appliance power
        """
        print("Disaggregating with Denoising Autoencoder...")

        if num_samples is None:
            num_samples = len(mains)

        # Create windows
        X_windows = self._create_windows(mains['power'].values[:num_samples])

        predictions = {}

        for app_name in self.appliance_names:
            model_info = self.models[app_name]
            model = model_info['model']
            X_max = model_info['X_max']
            y_max = model_info['y_max']

            model.eval()

            # Normalize
            X_norm = X_windows / X_max if X_max > 0 else X_windows
            X_tensor = torch.FloatTensor(X_norm).to(self.device)

            with torch.no_grad():
                # Predict in batches
                all_preds = []

                for i in range(0, len(X_tensor), self.batch_size):
                    batch = X_tensor[i:i + self.batch_size]
                    batch_preds = model(batch)
                    all_preds.append(batch_preds.cpu().numpy())

                # Concatenate and denormalize
                y_pred_norm = np.concatenate(all_preds, axis=0)
                y_pred_windows = y_pred_norm * y_max

                # Convert windows back to samples (use center value)
                y_pred_flat = self._windows_to_samples(y_pred_windows, num_samples)

                # Ensure non-negative
                y_pred_flat = np.maximum(0, y_pred_flat)

                predictions[app_name] = pd.DataFrame(
                    {'power': y_pred_flat},
                    index=mains.index[:num_samples]
                )

            print(f"  {app_name}: Predicted {num_samples} samples")

        print("Disaggregation complete!")
        return predictions

    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Create sliding windows from time series

        Args:
            data: 1D array

        Returns:
            2D array of windows
        """
        n_windows = len(data) - self.window_size + 1

        if n_windows <= 0:
            # Not enough data
            return data.reshape(1, -1)

        windows = np.zeros((n_windows, self.window_size))

        for i in range(n_windows):
            windows[i] = data[i:i + self.window_size]

        return windows

    def _windows_to_samples(self, windows: np.ndarray, target_length: int) -> np.ndarray:
        """
        Convert windows back to sample-wise predictions

        Args:
            windows: Array of windows
            target_length: Desired output length

        Returns:
            1D array of predictions
        """
        n_windows = len(windows)
        window_size = windows.shape[1]

        # Use center value from each window
        center_idx = window_size // 2
        samples = windows[:, center_idx]

        # Pad or truncate to target length
        if len(samples) < target_length:
            samples = np.pad(samples, (0, target_length - len(samples)), mode='edge')
        else:
            samples = samples[:target_length]

        return samples
