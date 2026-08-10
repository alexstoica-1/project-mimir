"""Pooled PyTorch LSTM for ticker-isolated volatility sequences."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.ml.data import (
    DATE_COLUMN,
    DEFAULT_LOOKBACK,
    TARGET_COLUMN,
    TICKER_COLUMN,
    FeaturePreprocessor,
    inverse_target,
    make_sequence_arrays,
)


def _torch_modules() -> tuple[Any, Any, Any]:
    """Import PyTorch lazily with a useful setup error."""

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - setup failure
        raise RuntimeError("Install torch before training the LSTM") from exc
    return torch, nn, (DataLoader, TensorDataset)


class VolatilityLSTM:
    """Two-layer pooled LSTM with a positive-volatility regression head."""

    def __init__(
        self,
        input_size: int,
        *,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        torch, nn, _ = _torch_modules()

        class Network(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, inputs: Any) -> Any:
                outputs, _ = self.lstm(inputs)
                return self.head(outputs[:, -1, :]).squeeze(-1)

        self.torch = torch
        self.network = Network()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

    def forward(self, inputs: Any) -> Any:
        """Forward one batch of sequences."""

        return self.network(inputs)

    def __call__(self, inputs: Any) -> Any:
        """Delegate call syntax to the wrapped PyTorch network."""

        return self.forward(inputs)


@dataclass
class LSTMTrainingResult:
    """Trained LSTM and its preprocessing/training metadata."""

    model: VolatilityLSTM
    preprocessor: FeaturePreprocessor
    lookback: int
    device: str
    history: pd.DataFrame

    def predict(self, frame: pd.DataFrame, *, start_after: pd.Timestamp | None = None) -> tuple[np.ndarray, pd.DataFrame]:
        """Predict sequence targets and return metadata for the requested interval."""

        torch, _, _ = _torch_modules()
        x, _, metadata = make_sequence_arrays(
            frame,
            self.preprocessor,
            lookback=self.lookback,
            start_after=start_after,
        )
        if len(x) == 0:
            return np.empty(0, dtype=float), metadata
        self.model.network.eval()
        with torch.no_grad():
            values = self.model(
                torch.from_numpy(x).to(self.device)
            ).detach().cpu().numpy()
        return inverse_target(values), metadata

    def save(self, path: str | Path) -> Path:
        """Save model state and preprocessing state for inference."""

        import joblib
        import torch

        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.network.state_dict(),
                "input_size": self.model.input_size,
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "lookback": self.lookback,
                "device": self.device,
                "preprocessor": self.preprocessor,
            },
            artifact_path,
        )
        # Keep a human-readable training history beside the binary state.
        self.history.to_csv(artifact_path.with_suffix(".history.csv"), index=False)
        return artifact_path

    @classmethod
    def load(cls, path: str | Path) -> "LSTMTrainingResult":
        """Load a saved PyTorch LSTM artifact."""

        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = VolatilityLSTM(
            payload["input_size"],
            hidden_size=payload["hidden_size"],
            num_layers=payload["num_layers"],
            dropout=payload["dropout"],
        )
        model.network.load_state_dict(payload["state_dict"])
        return cls(
            model=model,
            preprocessor=payload["preprocessor"],
            lookback=payload["lookback"],
            device="cpu",
            history=pd.read_csv(Path(path).with_suffix(".history.csv"))
            if Path(path).with_suffix(".history.csv").exists()
            else pd.DataFrame(),
        )


def train_lstm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    full_frame: pd.DataFrame,
    *,
    train_end: pd.Timestamp,
    validation_end: pd.Timestamp,
    lookback: int = DEFAULT_LOOKBACK,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    patience: int = 5,
    seed: int = 42,
) -> LSTMTrainingResult:
    """Train a pooled LSTM with train-only scaling and chronological early stopping."""

    torch, nn, data_modules = _torch_modules()
    DataLoader, TensorDataset = data_modules
    torch.set_num_threads(int(os.environ.get("MIMIR_TORCH_THREADS", "1")))
    torch.manual_seed(seed)
    np.random.seed(seed)

    preprocessor = FeaturePreprocessor.fit(train)
    train_x, train_y, _ = make_sequence_arrays(
        full_frame,
        preprocessor,
        lookback=lookback,
        end_at=train_end,
    )
    validation_x, validation_y, _ = make_sequence_arrays(
        full_frame,
        preprocessor,
        lookback=lookback,
        start_after=train_end,
        end_at=validation_end,
    )
    if len(train_x) == 0 or len(validation_x) == 0:
        raise ValueError("Not enough rows to construct LSTM train/validation sequences")

    # CPU is the deterministic default. MPS can be opted into explicitly,
    # which avoids backend initialization surprises on different macOS builds.
    requested_device = os.environ.get("MIMIR_TORCH_DEVICE", "cpu")
    if requested_device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = VolatilityLSTM(
        train_x.shape[-1],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model.network.to(device)
    optimizer = torch.optim.Adam(model.network.parameters(), lr=learning_rate)
    loss_function = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
    )
    validation_inputs = torch.from_numpy(validation_x).to(device)
    validation_targets = torch.from_numpy(validation_y).to(device)
    history: list[dict[str, float]] = []
    best_state = None
    best_validation_loss = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.network.train()
        train_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.network.eval()
        with torch.no_grad():
            validation_loss = float(
                loss_function(model(validation_inputs), validation_targets).detach().cpu()
            )
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.network.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.network.load_state_dict(best_state)
    return LSTMTrainingResult(
        model=model,
        preprocessor=preprocessor,
        lookback=lookback,
        device=device,
        history=pd.DataFrame(history),
    )
