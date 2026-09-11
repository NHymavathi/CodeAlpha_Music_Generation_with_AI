import os
import logging
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional

from src.dataset import load_processed_data
from src.model import load_trained_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_cached_model: Optional[tf.keras.Model] = None
_cached_model_mtime: float = 0.0

def get_model_instance(model_path: str) -> Optional[tf.keras.Model]:
    """
    Returns cached model instance or loads model from disk if updated.
    """
    global _cached_model, _cached_model_mtime
    if not os.path.exists(model_path):
        return None

    current_mtime = os.path.getmtime(model_path)
    if _cached_model is None or current_mtime > _cached_model_mtime:
        logger.info(f"Loading/reloading model from {model_path}...")
        _cached_model = load_trained_model(model_path)
        _cached_model_mtime = current_mtime

    return _cached_model

def sample_with_temperature(predictions: np.ndarray, temperature: float = 1.0) -> int:
    """
    Applies temperature scaling to output probability predictions and samples an index.
    """
    predictions = np.asarray(predictions).astype('float64')
    if temperature <= 0:
        return int(np.argmax(predictions))

    # Logarithm of predictions with numerical stability offset
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    
    # Sample index using multinomial distribution
    probabilities = np.random.multinomial(1, predictions, 1)
    return int(np.argmax(probabilities))

def generate_music_sequence(
    model_path: str,
    processed_dir: str,
    num_notes: int = 100,
    temperature: float = 1.0
) -> List[str]:
    data = load_processed_data(processed_dir)
    X = data['X']
    int_to_note = data['int_to_note']
    vocab_size = data['vocab_size']
    pitch_names = data.get('pitch_names', [])

    model = get_model_instance(model_path)
    if model is None:
        raise ValueError(f"Could not load trained model from {model_path}. Please train the model first.")

    try:
        sequence_length = model.input_shape[1] or data['sequence_length']
    except Exception:
        sequence_length = data['sequence_length']

    # Select random seed sequence from training dataset X
    start_index = np.random.randint(0, len(X) - 1)
    # Ensure seed length matches expected input sequence length
    pattern = list(X[start_index][:sequence_length])
    if len(pattern) < sequence_length:
        pattern = list(np.pad(pattern, (sequence_length - len(pattern), 0), 'constant'))

    generated_notes = []
    logger.info(f"Generating {num_notes} notes with sequence_length={sequence_length}, temperature={temperature}...")

    for i in range(num_notes):
        prediction_input = np.reshape(pattern, (1, len(pattern)))
        # Fast direct forward pass in tensor mode
        prediction_tensor = model(prediction_input, training=False)
        prediction = prediction_tensor.numpy()[0]

        index = sample_with_temperature(prediction, temperature=temperature)
        result = int_to_note.get(index, pitch_names[0] if pitch_names else "C4")
        generated_notes.append(result)

        pattern.append(index)
        pattern = pattern[1:]

    logger.info(f"Successfully generated {len(generated_notes)} notes/chords.")
    return generated_notes
