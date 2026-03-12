"""
models/fraud_detection/autoencoder.py
---------------------------------------
PyTorch Autoencoder for anomaly-based fraud detection.

Architecture
------------
An autoencoder consists of two symmetric networks:

    Input  ──► Encoder ──► Latent (bottleneck) ──► Decoder ──► Reconstruction

Encoder: compresses the input into a *low-dimensional* latent representation.
         This forces the network to capture only the most important patterns.

Decoder: reconstructs the original input from the latent representation.

Training signal: Mean-Squared-Error (MSE) between input and reconstruction.

Anomaly detection logic
------------------------
The autoencoder is trained **only on normal (non-fraudulent) data** (or the
full dataset if labels are unavailable – anomalies are then rare enough to
remain poorly reconstructed).

After training:
  * Normal samples → low reconstruction error (the model has seen them often).
  * Anomalous samples → high reconstruction error (outside the learned manifold).

A threshold (e.g. 95th percentile of training errors) separates normal from
fraudulent applications.

Architecture details
---------------------
    Input dim  →  128  →  64  →  32  (latent)  →  64  →  128  →  Input dim
    Activation: ReLU for hidden layers; Sigmoid for the final output layer
                (works best when input is normalised to [0, 1]; if using
                 StandardScaler keep the output activation as linear/identity).

    BatchNorm after each hidden layer stabilises training on financial data
    which can have highly varying feature magnitudes.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class LoanDataset(Dataset):
    """
    Simple PyTorch Dataset that wraps a numpy feature matrix.

    For an autoencoder the *target* is the input itself (reconstruction task).

    Parameters
    ----------
    X : np.ndarray  shape (n_samples, n_features)  – must be float32
    """

    def __init__(self, X: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # Both input and target are the same sample (reconstruction)
        return self.X[idx], self.X[idx]


# ---------------------------------------------------------------------------
# Autoencoder model
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """
    Symmetric encoder–decoder with BatchNorm and ReLU activations.

    Parameters
    ----------
    input_dim  : int   Number of input features.
    latent_dim : int   Size of the bottleneck (latent) representation.
                       Smaller → stronger compression → more sensitive anomaly
                       detector, but also harder to train.
    """

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder: input_dim → 64 → 32 → latent_dim
        # ------------------------------------------------------------------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

        # ------------------------------------------------------------------
        # Decoder: latent_dim → 32 → 64 → input_dim
        # ------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            # No activation: output is in the same space as the StandardScaler
            # output (approximately Gaussian), so linear output is appropriate.
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then decode the input."""
        z = self.encoder(x)       # compress to latent space
        x_recon = self.decoder(z) # reconstruct from latent space
        return x_recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return only the latent representation (useful for visualisation)."""
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Training and inference wrapper
# ---------------------------------------------------------------------------

class AutoencoderDetector:
    """
    Full pipeline: build, train, threshold, predict, and evaluate.

    Parameters
    ----------
    input_dim   : int   Number of input features (must match training data).
    latent_dim  : int   Bottleneck size.
    lr          : float Learning rate for Adam optimiser.
    batch_size  : int
    epochs      : int   Maximum training epochs.
    patience    : int   Early stopping patience (epochs without validation
                        loss improvement before stopping).
    device      : str   'cuda' if GPU available, else 'cpu'.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 8,
        lr: float = 1e-3,
        batch_size: int = 2048,
        epochs: int = 50,
        patience: int = 10,
        device: str | None = None,
        max_train_samples: int | None = None,
    ):
        if device is not None:
            self.device = device
        elif torch.backends.mps.is_available():
            self.device = "mps"      # Apple Silicon GPU
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.max_train_samples = max_train_samples
        self.model = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.threshold: float | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray | None = None,
        threshold_percentile: float = 95.0,
    ) -> dict:
        """
        Train the autoencoder and compute the anomaly threshold.

        Training strategy
        -----------------
        1. Build DataLoaders for train (and optional validation) sets.
        2. Adam optimiser with MSELoss (reconstruction loss).
        3. Early stopping monitors validation loss to prevent overfitting.
        4. After training, compute per-sample reconstruction error on X_train.
        5. Set threshold as the *threshold_percentile* of those errors.
           e.g. 95th percentile → ~5 % of training samples are flagged, which
           roughly corresponds to *contamination = 0.05*.

        Parameters
        ----------
        X_train              : numpy float32 array
        X_val                : optional validation set (same format)
        threshold_percentile : percentile of training errors to use as cut-off

        Returns
        -------
        dict  training history {'train_loss': [...], 'val_loss': [...]}
        """
        # Optionally subsample to keep epoch time manageable on large datasets
        if self.max_train_samples and len(X_train) > self.max_train_samples:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(X_train), self.max_train_samples, replace=False)
            X_train = X_train[idx]
            print(f"[Autoencoder] Subsampled to {self.max_train_samples:,} rows for training.")

        print(f"[Autoencoder] Training on {X_train.shape[0]:,} samples, "
              f"device={self.device} …")

        # MPS doesn't benefit from multi-process data loading; CPU benefits from 2 workers
        nw = 0 if self.device == "mps" else 2
        train_loader = DataLoader(
            LoanDataset(X_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=nw,
            pin_memory=(self.device == "cuda"),
            persistent_workers=(nw > 0),
        )
        val_loader = (
            DataLoader(
                LoanDataset(X_val),
                batch_size=self.batch_size * 2,
                shuffle=False,
                num_workers=nw,
                pin_memory=(self.device == "cuda"),
                persistent_workers=(nw > 0),
            )
            if X_val is not None else None
        )

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        history = {"train_loss": [], "val_loss": []}

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_state = None

        for epoch in range(1, self.epochs + 1):
            # ---- Training pass ----
            self.model.train()
            train_losses = []
            for x_batch, _ in train_loader:
                x_batch = x_batch.to(self.device)
                optimiser.zero_grad()
                recon = self.model(x_batch)
                loss = criterion(recon, x_batch)
                loss.backward()
                optimiser.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            history["train_loss"].append(avg_train)

            # ---- Validation pass ----
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for x_val, _ in val_loader:
                        x_val = x_val.to(self.device)
                        recon = self.model(x_val)
                        val_losses.append(criterion(recon, x_val).item())
                avg_val = np.mean(val_losses)
                history["val_loss"].append(avg_val)

                # Early stopping
                if avg_val < best_val_loss - 1e-6:
                    best_val_loss = avg_val
                    epochs_no_improve = 0
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if epoch % 10 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:3d}/{self.epochs} | "
                          f"train={avg_train:.6f}  val={avg_val:.6f}")

                if epochs_no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break
            else:
                if epoch % 10 == 0 or epoch == 1:
                    print(f"  Epoch {epoch:3d}/{self.epochs} | train={avg_train:.6f}")

        # Restore best weights (if validation was used)
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # ---- Set anomaly threshold ----
        train_errors = self._reconstruction_errors(X_train)
        self.threshold = float(np.percentile(train_errors, threshold_percentile))
        self._is_fitted = True
        print(f"[Autoencoder] Threshold set at {threshold_percentile}th percentile "
              f"= {self.threshold:.6f}")
        return history

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Compute per-sample MSE reconstruction error (public API)."""
        self._check_fitted()
        return self._reconstruction_errors(X)

    def _reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Internal method – works even before _is_fitted is set."""
        self.model.eval()
        dataset = LoanDataset(X)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        errors = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(self.device)
                recon = self.model(x_batch)
                # Per-sample MSE: mean over feature dimension
                mse = ((recon - x_batch) ** 2).mean(dim=1)
                errors.append(mse.cpu().numpy())
        return np.concatenate(errors)

    def flag_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Return boolean array: True = sample is an anomaly.

        Samples whose reconstruction error exceeds *self.threshold* are
        considered fraudulent / suspicious.
        """
        self._check_fitted()
        errors = self._reconstruction_errors(X)
        return errors > self.threshold

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 (anomaly) or 0 (normal) for each sample."""
        return self.flag_anomalies(X).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        fraud_label: int = 1,
    ) -> dict:
        """
        Evaluate the detector against ground-truth labels.

        Parameters
        ----------
        X          : scaled feature matrix
        y_true     : integer array with fraud_label for fraud, else normal
        fraud_label: which value in y_true represents fraud
        """
        from sklearn.metrics import (
            precision_score, recall_score, f1_score,
            roc_auc_score, classification_report,
        )
        self._check_fitted()
        errors = self._reconstruction_errors(X)
        y_pred_binary = (errors > self.threshold).astype(int)
        y_true_binary = (y_true == fraud_label).astype(int)

        auc = roc_auc_score(y_true_binary, errors)   # higher error → more anomalous

        metrics = {
            "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
            "recall":    recall_score(y_true_binary, y_pred_binary, zero_division=0),
            "f1":        f1_score(y_true_binary, y_pred_binary, zero_division=0),
            "roc_auc":   auc,
        }
        print("\n[Autoencoder] Evaluation results:")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v:.4f}")
        print("\n" + classification_report(
            y_true_binary, y_pred_binary,
            target_names=["Normal", "Fraud"], zero_division=0,
        ))
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str, meta_path: str | None = None) -> None:
        """
        Save model weights (PyTorch .pt) and threshold metadata (joblib).

        Parameters
        ----------
        model_path : path for the PyTorch state-dict (.pt file)
        meta_path  : optional path for threshold metadata (.joblib)
        """
        torch.save(self.model.state_dict(), model_path)
        if meta_path:
            joblib.dump({"threshold": self.threshold}, meta_path)
        print(f"[Autoencoder] Weights saved to '{model_path}'.")

    @classmethod
    def load(
        cls,
        model_path: str,
        input_dim: int,
        latent_dim: int = 8,
        meta_path: str | None = None,
        device: str | None = None,
    ) -> "AutoencoderDetector":
        """Load a previously saved autoencoder."""
        detector = cls(
            input_dim=input_dim,
            latent_dim=latent_dim,
            device=device,
        )
        state = torch.load(model_path, map_location=detector.device)
        detector.model.load_state_dict(state)
        if meta_path:
            meta = joblib.load(meta_path)
            detector.threshold = meta["threshold"]
        detector._is_fitted = True
        print(f"[Autoencoder] Loaded from '{model_path}'.")
        return detector

    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "Autoencoder is not fitted yet. Call .fit(X) first."
            )
