import logging
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_lstm_model(
    vocab_size: int,
    sequence_length: int = 30,
    embedding_dim: int = 64,
    lstm_units: int = 128,
    dropout: float = 0.2,
    learning_rate: float = 0.001
) -> Sequential:
    """
    Builds and compiles the Keras LSTM Model for Music Generation.
    """
    model = Sequential([
        Input(shape=(sequence_length,), name="input_sequence"),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, name="embedding"),
        LSTM(lstm_units, return_sequences=True, name="lstm_1"),
        Dropout(dropout, name="dropout_1"),
        LSTM(lstm_units, name="lstm_2"),
        Dropout(dropout, name="dropout_2"),
        Dense(256, activation='relu', name="dense_1"),
        Dense(vocab_size, activation='softmax', name="output_probabilities")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    logger.info(f"Built LSTM model with vocab_size={vocab_size}, seq_length={sequence_length}, units={lstm_units}")
    model.summary(print_fn=lambda x: logger.info(x))
    return model

def load_trained_model(model_path: str) -> Optional[Sequential]:
    """
    Loads a saved Keras model from disk.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None
