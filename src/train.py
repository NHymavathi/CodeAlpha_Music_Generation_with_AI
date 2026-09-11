import os
import json
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from typing import Dict, Any, Optional

from src.dataset import load_processed_data
from src.model import build_lstm_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to store training status in-memory for fast API access
training_status: Dict[str, Any] = {
    "is_training": False,
    "current_epoch": 0,
    "total_epochs": 0,
    "loss": [],
    "val_loss": [],
    "accuracy": [],
    "val_accuracy": [],
    "message": "Model not trained yet."
}

class TrainingProgressCallback(Callback):
    """
    Custom Keras callback to capture epoch metrics for real-time Web UI progress updates.
    """
    def __init__(self, history_save_path: str, total_epochs: int):
        super().__init__()
        self.history_save_path = history_save_path
        self.total_epochs = total_epochs

    def on_train_begin(self, logs=None):
        global training_status
        training_status["is_training"] = True
        training_status["current_epoch"] = 0
        training_status["total_epochs"] = self.total_epochs
        training_status["loss"] = []
        training_status["val_loss"] = []
        training_status["accuracy"] = []
        training_status["val_accuracy"] = []
        training_status["message"] = "Training started..."
        self._save_status()

    def on_epoch_end(self, epoch, logs=None):
        global training_status
        logs = logs or {}
        training_status["current_epoch"] = epoch + 1
        training_status["loss"].append(float(logs.get("loss", 0)))
        training_status["val_loss"].append(float(logs.get("val_loss", 0)))
        training_status["accuracy"].append(float(logs.get("accuracy", 0)))
        training_status["val_accuracy"].append(float(logs.get("val_accuracy", 0)))
        training_status["message"] = f"Epoch {epoch + 1}/{self.total_epochs} completed. Loss: {logs.get('loss', 0):.4f}, Acc: {logs.get('accuracy', 0):.4f}"
        self._save_status()
        logger.info(training_status["message"])

    def on_train_end(self, logs=None):
        global training_status
        training_status["is_training"] = False
        training_status["message"] = "Training completed successfully!"
        self._save_status()

    def _save_status(self):
        try:
            os.makedirs(os.path.dirname(self.history_save_path), exist_ok=True)
            with open(self.history_save_path, 'w') as f:
                json.dump(training_status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write history status file: {e}")

def get_training_status() -> Dict[str, Any]:
    """Returns current in-memory training status."""
    return training_status

def train_model(
    processed_dir: str,
    model_save_path: str,
    history_save_path: str,
    sequence_length: int = 30,
    embedding_dim: int = 64,
    lstm_units: int = 128,
    dropout: float = 0.2,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    val_split: float = 0.2
) -> Dict[str, Any]:
    """
    Loads dataset, splits into train/val, builds model, runs training, and saves model + history.
    """
    global training_status
    data = load_processed_data(processed_dir)
    X = data['X']
    y = data['y']
    vocab_size = data['vocab_size']

    # Shuffle and split train / validation
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    val_size = int(num_samples * val_split)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    if len(val_indices) == 0:
        X_train, y_train = X, y
        X_val, y_val = X, y
    else:
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]

    # Build model
    model = build_lstm_model(
        vocab_size=vocab_size,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout=dropout,
        learning_rate=learning_rate
    )

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_loss' if len(val_indices) > 0 else 'loss',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss' if len(val_indices) > 0 else 'loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        TrainingProgressCallback(history_save_path, total_epochs=epochs)
    ]

    logger.info(f"Starting training on {len(X_train)} samples, validation on {len(X_val)} samples for {epochs} epochs...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(val_indices) > 0 else None,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Ensure model is saved even if EarlyStopping didn't save
    model.save(model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    return training_status
