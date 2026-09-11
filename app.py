import os
import json
import threading
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import config
from src.midi_processor import parse_midi_dataset, find_midi_files
from src.dataset import create_sequences, save_processed_data, load_processed_data
from src.train import train_model, get_training_status
from src.generate import generate_music_sequence
from src.midi_generator import create_midi_from_notes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ai-music-generation-lstm-secret-key'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

# Ensure directories exist
for folder in [config.RAW_DATA_DIR, config.PROCESSED_DATA_DIR, config.GENERATED_DATA_DIR, config.MODELS_DIR, config.UPLOADS_DIR]:
    os.makedirs(folder, exist_ok=True)

def is_allowed_file(filename: str) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    return ext in config.ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Returns dataset summary statistics and raw file list."""
    try:
        midi_files = find_midi_files(config.RAW_DATA_DIR)
        file_list = [os.path.basename(f) for f in midi_files]
        
        is_preprocessed = os.path.exists(config.NOTES_DATA_PATH)
        processed_info = {}
        
        if is_preprocessed:
            try:
                data = load_processed_data(config.PROCESSED_DATA_DIR)
                processed_info = {
                    "num_samples": data["num_samples"],
                    "vocab_size": data["vocab_size"],
                    "sequence_length": data["sequence_length"]
                }
            except Exception as e:
                logger.warning(f"Failed to load processed dataset metadata: {e}")
                is_preprocessed = False

        model_exists = os.path.exists(config.MODEL_SAVE_PATH)

        return jsonify({
            "success": True,
            "num_files": len(midi_files),
            "files": file_list,
            "is_preprocessed": is_preprocessed,
            "processed_info": processed_info,
            "model_exists": model_exists
        })
    except Exception as e:
        logger.error(f"Error fetching dataset info: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_midi():
    """Uploads MIDI file(s) into raw dataset directory."""
    try:
        if 'file' not in request.files:
            return jsonify({"success": False, "error": "No file part in request"}), 400

        files = request.files.getlist('file')
        uploaded_files = []

        for file in files:
            if file and file.filename != '':
                if not is_allowed_file(file.filename):
                    return jsonify({"success": False, "error": f"Invalid file format: '{file.filename}'. Only .mid or .midi allowed."}), 400

                filename = secure_filename(file.filename)
                save_path = os.path.join(config.RAW_DATA_DIR, filename)
                file.save(save_path)
                uploaded_files.append(filename)
                logger.info(f"Uploaded MIDI file saved to {save_path}")

        return jsonify({
            "success": True,
            "message": f"Successfully uploaded {len(uploaded_files)} file(s).",
            "uploaded_files": uploaded_files
        })
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/preprocess', methods=['POST'])
def preprocess_dataset():
    """Parses raw MIDI files, extracts pitch sequences, and saves processed data."""
    try:
        req_data = request.get_json() or {}
        sequence_length = int(req_data.get('sequence_length', config.SEQUENCE_LENGTH))

        notes, stats = parse_midi_dataset(config.RAW_DATA_DIR)

        if not notes:
            return jsonify({"success": False, "error": "No notes found in raw MIDI dataset. Please upload or add MIDI files first."}), 400

        if len(notes) <= sequence_length:
            return jsonify({"success": False, "error": f"Extracted only {len(notes)} notes, which is less than sequence length {sequence_length}. Please add more MIDI files or lower sequence length."}), 400

        X, y, note_to_int, int_to_note, pitch_names = create_sequences(notes, sequence_length=sequence_length)

        save_processed_data(
            config.PROCESSED_DATA_DIR,
            X, y, note_to_int, int_to_note, pitch_names,
            sequence_length=sequence_length
        )

        return jsonify({
            "success": True,
            "message": "Dataset preprocessed successfully!",
            "stats": stats,
            "num_samples": len(X),
            "vocab_size": len(pitch_names),
            "sequence_length": sequence_length
        })
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/train', methods=['POST'])
def start_training():
    """Triggers LSTM model training asynchronously in a background thread."""
    try:
        status = get_training_status()
        if status.get("is_training", False):
            return jsonify({"success": False, "error": "Model training is already in progress!"}), 400

        if not os.path.exists(config.NOTES_DATA_PATH):
            return jsonify({"success": False, "error": "Processed dataset not found. Please click 'Preprocess Dataset' first."}), 400

        req = request.get_json() or {}
        sequence_length = int(req.get('sequence_length', config.SEQUENCE_LENGTH))
        embedding_dim = int(req.get('embedding_dim', config.EMBEDDING_DIM))
        lstm_units = int(req.get('lstm_units', config.LSTM_UNITS))
        dropout = float(req.get('dropout', config.DROPOUT))
        epochs = int(req.get('epochs', config.EPOCHS))
        batch_size = int(req.get('batch_size', config.BATCH_SIZE))
        learning_rate = float(req.get('learning_rate', config.LEARNING_RATE))

        # Launch training in daemon thread
        def run_train():
            try:
                train_model(
                    processed_dir=config.PROCESSED_DATA_DIR,
                    model_save_path=config.MODEL_SAVE_PATH,
                    history_save_path=config.HISTORY_SAVE_PATH,
                    sequence_length=sequence_length,
                    embedding_dim=embedding_dim,
                    lstm_units=lstm_units,
                    dropout=dropout,
                    epochs=epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate
                )
            except Exception as ex:
                logger.error(f"Async training failed: {ex}")
                status["is_training"] = False
                status["message"] = f"Training failed: {ex}"

        t = threading.Thread(target=run_train, daemon=True)
        t.start()

        return jsonify({
            "success": True,
            "message": "Training started successfully in the background!"
        })
    except Exception as e:
        logger.error(f"Error initiating training: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/train-status', methods=['GET'])
def train_status():
    """Returns training progress and history for UI polling."""
    try:
        status = get_training_status()
        
        # Check if saved history file exists on disk
        history = {}
        if os.path.exists(config.HISTORY_SAVE_PATH):
            try:
                with open(config.HISTORY_SAVE_PATH, 'r') as f:
                    history = json.load(f)
            except Exception:
                pass

        model_exists = os.path.exists(config.MODEL_SAVE_PATH)

        return jsonify({
            "success": True,
            "status": status,
            "saved_history": history,
            "model_exists": model_exists
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate_music():
    """Generates new music sequence with LSTM model and saves to MIDI file."""
    try:
        if not os.path.exists(config.MODEL_SAVE_PATH):
            return jsonify({"success": False, "error": "No trained model found! Please train the model first."}), 400

        req = request.get_json() or {}
        num_notes = int(req.get('num_notes', config.NUM_GENERATE_NOTES))
        temperature = float(req.get('temperature', config.TEMPERATURE))

        notes = generate_music_sequence(
            model_path=config.MODEL_SAVE_PATH,
            processed_dir=config.PROCESSED_DATA_DIR,
            num_notes=num_notes,
            temperature=temperature
        )

        # Create unique filename
        filename = f"generated_music_{len(os.listdir(config.GENERATED_DATA_DIR)) + 1:03d}.mid"
        output_filepath = os.path.join(config.GENERATED_DATA_DIR, filename)

        create_midi_from_notes(notes, output_filepath)

        return jsonify({
            "success": True,
            "message": "Music generated successfully!",
            "filename": filename,
            "num_notes": len(notes),
            "temperature": temperature,
            "notes": notes[:50]  # First 50 generated notes/chords for visualization preview
        })
    except Exception as e:
        logger.error(f"Error generating music: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename: str):
    """Download generated MIDI file."""
    return send_from_directory(config.GENERATED_DATA_DIR, filename, as_attachment=True)

@app.route('/play/<filename>', methods=['GET'])
def play_file(filename: str):
    """Stream generated MIDI file for Web Audio API playback."""
    return send_from_directory(config.GENERATED_DATA_DIR, filename, mimetype='audio/midi')

if __name__ == '__main__':
    logger.info("Starting AI Music Generation Flask Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)
