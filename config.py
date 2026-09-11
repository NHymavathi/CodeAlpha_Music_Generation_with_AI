import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
GENERATED_DATA_DIR = os.path.join(DATA_DIR, 'generated')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model file path
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, 'music_lstm.h5')
HISTORY_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, 'history.json')
NOTES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, 'notes_data.pkl')

# Hyperparameters / Defaults
SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 64
LSTM_UNITS = 128
DROPOUT = 0.2
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TEMPERATURE = 1.0
NUM_GENERATE_NOTES = 100

# Allowed MIDI extensions
ALLOWED_EXTENSIONS = {'.mid', '.midi'}
