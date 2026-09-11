import os
import pickle
import numpy as np
import logging
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sequences(notes: List[str], sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, Dict[str, int], Dict[int, str], List[str]]:
    """
    Creates input (X) and target (y) integer sequences from extracted notes list.
    """
    if len(notes) <= sequence_length:
        raise ValueError(f"Extracted dataset has only {len(notes)} notes, which is less than sequence length {sequence_length}. Please add more MIDI files or reduce sequence length.")

    # Extract sorted set of unique pitch names
    pitch_names = sorted(list(set(notes)))
    vocab_size = len(pitch_names)

    # Create mapping dictionaries
    note_to_int = {note: number for number, note in enumerate(pitch_names)}
    int_to_note = {number: note for number, note in enumerate(pitch_names)}

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    X = np.array(network_input, dtype=np.int32)
    y = np.array(network_output, dtype=np.int32)

    logger.info(f"Created {len(X)} training sequences of length {sequence_length}. Vocab size: {vocab_size}")
    return X, y, note_to_int, int_to_note, pitch_names

def save_processed_data(processed_dir: str, X: np.ndarray, y: np.ndarray, note_to_int: Dict[str, int], int_to_note: Dict[int, str], pitch_names: List[str], sequence_length: int) -> str:
    """
    Saves processed dataset arrays and mappings to pickle file.
    """
    os.makedirs(processed_dir, exist_ok=True)
    save_path = os.path.join(processed_dir, 'notes_data.pkl')

    data = {
        'X': X,
        'y': y,
        'note_to_int': note_to_int,
        'int_to_note': int_to_note,
        'pitch_names': pitch_names,
        'vocab_size': len(pitch_names),
        'sequence_length': sequence_length,
        'num_samples': len(X)
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    logger.info(f"Processed dataset saved successfully to {save_path}")
    return save_path

def load_processed_data(processed_dir: str) -> Dict[str, Any]:
    """
    Loads preprocessed dataset dictionary from pickle file.
    """
    save_path = os.path.join(processed_dir, 'notes_data.pkl')
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"Processed dataset file not found at {save_path}. Please run preprocessing first.")

    with open(save_path, 'rb') as f:
        data = pickle.load(f)

    logger.info(f"Loaded processed dataset from {save_path}. Samples: {data['num_samples']}, Vocab size: {data['vocab_size']}")
    return data
