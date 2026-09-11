import os
from src.midi_processor import parse_midi_dataset
from src.dataset import create_sequences, save_processed_data, load_processed_data
from src.model import build_lstm_model
import config

def test_pipeline():
    print("=== TEST STEP 1: Parse MIDI Dataset ===")
    notes, stats = parse_midi_dataset(config.RAW_DATA_DIR)
    print("Stats:", stats)
    assert len(notes) > 0, "Notes list should not be empty!"

    print("\n=== TEST STEP 2: Create Sequences ===")
    seq_len = 15
    X, y, note_to_int, int_to_note, pitch_names = create_sequences(notes, sequence_length=seq_len)
    print(f"X shape: {X.shape}, y shape: {y.shape}, Vocab size: {len(pitch_names)}")

    print("\n=== TEST STEP 3: Save & Load Processed Data ===")
    save_path = save_processed_data(
        config.PROCESSED_DATA_DIR, X, y, note_to_int, int_to_note, pitch_names, sequence_length=seq_len
    )
    loaded_data = load_processed_data(config.PROCESSED_DATA_DIR)
    assert loaded_data['vocab_size'] == len(pitch_names)

    print("\n=== TEST STEP 4: Build Model ===")
    model = build_lstm_model(
        vocab_size=loaded_data['vocab_size'],
        sequence_length=seq_len,
        embedding_dim=32,
        lstm_units=64
    )
    assert model is not None, "Model creation failed!"
    print("Model summary built successfully!")

if __name__ == '__main__':
    test_pipeline()
