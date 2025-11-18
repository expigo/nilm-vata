"""
Sequence-to-Sequence LSTM algorithm for NILM

Deep learning approach using LSTM encoder-decoder architecture.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
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
        "Seq2Seq algorithm will not work without PyTorch."
    )
    # Create dummy classes to avoid import errors
    class nn:
        class Module:
            pass


if TORCH_AVAILABLE:
    class Seq2SeqLSTM(nn.Module):
        """LSTM Encoder-Decoder for NILM"""

        def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1):
            """
            Initialize Seq2Seq model

            Args:
                input_size: Input feature dimension
                hidden_size: LSTM hidden size
                num_layers: Number of LSTM layers
                output_size: Output dimension
            """
            super(Seq2SeqLSTM, self).__init__()

            self.hidden_size = hidden_size
            self.num_layers = num_layers

            # Encoder
            self.encoder = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )

            # Decoder
            self.decoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0
            )

            # Output layer
            self.fc = nn.Linear(hidden_size, output_size)
            self.relu = nn.ReLU()

        def forward(self, x):
            """
            Forward pass

            Args:
                x: Input sequence [batch, seq_len, input_size]

            Returns:
                Output sequence [batch, seq_len, output_size]
            """
            batch_size, seq_len, _ = x.shape

            # Encode
            encoder_out, (hidden, cell) = self.encoder(x)

            # Decode
            decoder_input = encoder_out
            decoder_out, _ = self.decoder(decoder_input, (hidden, cell))

            # Output
            out = self.fc(decoder_out)
            out = self.relu(out)

            return out
else:
    # Dummy class when PyTorch not available
    Seq2SeqLSTM = None


class Seq2SeqDisaggregator:
    """
    Sequence-to-Sequence disaggregator using LSTM

    Uses deep learning to learn temporal patterns in aggregate consumption
    and predict appliance-level consumption.
    """

    def __init__(
        self,
        sequence_length: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Initialize Seq2Seq disaggregator

        Args:
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for Seq2Seq. Install with: pip install torch"
            )

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.models = {}
        self.appliance_names = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, mains: pd.DataFrame, appliances: Dict[str, pd.DataFrame]):
        """
        Train Seq2Seq models for each appliance

        Args:
            mains: Aggregate power consumption data
            appliances: Dictionary of appliance-wise power data
        """
        print(f"Training Seq2Seq LSTM (device={self.device})...")
        print(f"  Sequence length: {self.sequence_length}")
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Epochs: {self.epochs}")

        self.appliance_names = list(appliances.keys())

        # Prepare sequences
        X_sequences = self._create_sequences(mains['power'].values)

        for app_name, app_data in appliances.items():
            print(f"\n  Training {app_name}...")

            y_sequences = self._create_sequences(app_data['power'].values)

            # Align lengths
            min_len = min(len(X_sequences), len(y_sequences))
            X = X_sequences[:min_len]
            y = y_sequences[:min_len]

            # Convert to tensors
            X_tensor = torch.FloatTensor(X).unsqueeze(-1).to(self.device)
            y_tensor = torch.FloatTensor(y).unsqueeze(-1).to(self.device)

            # Create dataset and dataloader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True
            )

            # Initialize model
            model = Seq2SeqLSTM(
                input_size=1,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_size=1
            ).to(self.device)

            # Loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

            # Training loop
            model.train()
            for epoch in range(self.epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    # Forward pass
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)

                if (epoch + 1) % 5 == 0:
                    print(f"    Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")

            self.models[app_name] = model
            print(f"    Training complete!")

    def disaggregate(self, mains: pd.DataFrame, num_samples: int = None) -> Dict[str, pd.DataFrame]:
        """
        Disaggregate using trained Seq2Seq models

        Args:
            mains: Aggregate power consumption data
            num_samples: Number of samples to process

        Returns:
            Dictionary of predicted appliance power consumption
        """
        print("Disaggregating with Seq2Seq LSTM...")

        if num_samples is None:
            num_samples = len(mains)

        # Create sequences
        X_sequences = self._create_sequences(mains['power'].values[:num_samples])
        X_tensor = torch.FloatTensor(X_sequences).unsqueeze(-1).to(self.device)

        predictions = {}

        for app_name in self.appliance_names:
            model = self.models[app_name]
            model.eval()

            with torch.no_grad():
                # Predict in batches
                all_preds = []

                for i in range(0, len(X_tensor), self.batch_size):
                    batch = X_tensor[i:i + self.batch_size]
                    batch_preds = model(batch)
                    all_preds.append(batch_preds.cpu().numpy())

                # Concatenate predictions
                y_pred = np.concatenate(all_preds, axis=0)

                # Flatten sequences back to samples
                y_pred_flat = y_pred.reshape(-1)

                # Pad to match num_samples
                if len(y_pred_flat) < num_samples:
                    y_pred_flat = np.pad(
                        y_pred_flat,
                        (0, num_samples - len(y_pred_flat)),
                        mode='edge'
                    )
                else:
                    y_pred_flat = y_pred_flat[:num_samples]

                # Ensure non-negative
                y_pred_flat = np.maximum(0, y_pred_flat)

                predictions[app_name] = pd.DataFrame(
                    {'power': y_pred_flat},
                    index=mains.index[:num_samples]
                )

            print(f"  {app_name}: Predicted {num_samples} samples")

        print("Disaggregation complete!")
        return predictions

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Create sequences from time series data

        Args:
            data: 1D array of power values

        Returns:
            2D array of sequences [num_sequences, sequence_length]
        """
        n_sequences = len(data) - self.sequence_length + 1

        if n_sequences <= 0:
            # Not enough data, return padded
            return data.reshape(1, -1)

        sequences = np.zeros((n_sequences, self.sequence_length))

        for i in range(n_sequences):
            sequences[i] = data[i:i + self.sequence_length]

        return sequences
