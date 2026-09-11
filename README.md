# AI Music Generation using LSTM 🎵

An end-to-end Deep Learning web application that learns musical patterns from MIDI datasets (classical compositions preferred) using a Recurrent Neural Network with Long Short-Term Memory (LSTM) layers, generates original musical note sequences using temperature-based sampling, and converts them back into playable `.mid` MIDI files with browser Web Audio synthesis.

---

## 📌 Project Overview & Pipeline

```text
  Raw MIDI Files (.mid/.midi)
             │
             ▼
  Music21 MIDI Parser (Notes & Chords Extraction)
             │
             ▼
  Sequence Creation & Integer Mapping (Sliding Window X, y)
             │
             ▼
  LSTM Neural Network Model (Embedding → LSTM → Dropout → Dense)
             │
             ▼
  Model Training (Sparse Categorical Crossentropy, Adam, Checkpoints)
             │
             ▼
  Temperature-based Next Note Prediction (Predictable / Balanced / Experimental)
             │
             ▼
  Music21 Stream Conversion & MIDI Export (data/generated/generated_music_XXX.mid)
             │
             ▼
  Browser Web Audio Synthesis (Tone.js) & MIDI Download
```

---

## ✨ Features

- **MIDI Dataset Ingestion & Upload**: Recursively scans `data/raw/` and supports drag-and-drop uploading of `.mid` / `.midi` files with format validation.
- **Robust MIDI Preprocessing**: Parses single pitches (`C4`, `F#3`) and polyphonic chords (`C4.E4.G4`) into standard string tokens using `music21`.
- **Customizable LSTM Model**: Adjustable sequence length, embedding dimension, LSTM hidden units, dropout, epochs, batch size, and learning rate.
- **Real-Time Training Analytics**: Asynchronous background training executor with real-time Loss & Accuracy metrics charts powered by Chart.js.
- **Temperature-Based Music Sampling**: Implements temperature scaling ($0.2 \le T \le 2.0$) to control creativity vs predictability during note generation.
- **In-Browser Audio Synthesizer**: Plays generated MIDI files directly in the browser using Tone.js Web Audio synthesis, with Play/Stop controls and pitch roll visualization.
- **MIDI File Export & Download**: Instantly saves valid, standard `.mid` files to `data/generated/` ready to download and open in any Digital Audio Workstation (DAW) or media player.

---

## 🛠️ Technology Stack

- **Backend Framework**: Python 3.10+, Flask
- **Deep Learning / AI**: TensorFlow 2.10+, Keras, NumPy (<2.0.0)
- **Music Processing**: `music21`
- **Frontend UI**: HTML5, CSS3 (Glassmorphism), JavaScript (ES6+), Bootstrap 5
- **Visualizations**: Chart.js
- **Audio Synthesizer**: Tone.js, @tonejs/midi

---

## 🚀 Installation & Setup

### 1. Clone or Open Project Directory

```bash
cd task-3
```

### 2. Create and Activate Virtual Environment (Recommended)

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🎼 Adding MIDI Data

Place raw `.mid` or `.midi` files into the `data/raw/` directory.

The application comes pre-populated with 4 public-domain classical MIDI samples (`sample_bach_invention.mid`, `sample_beethoven_theme.mid`, `sample_mozart_minuet.mid`, `sample_chopin_prelude.mid`).

Alternatively, you can drag and drop your own MIDI files directly into the web interface.

---

## 🏃 Running the Application

Start the Flask development server:

```bash
python app.py
```

Then open your browser and navigate to:

```text
http://127.0.0.1:5000
```

---

## 🧠 How the LSTM Model Works

1. **Tokenization**: Musical notes and chords extracted by `music21` are mapped to unique integer IDs ($0 \dots V-1$).
2. **Sliding Window Sequences**: The input data $X$ consists of note index windows of fixed length (e.g., 30 notes), and target $y$ is the immediate next note index.
3. **Embedding Layer**: Projects categorical note integers into a dense continuous vector space where musically similar pitches reside close together.
4. **Stacked LSTM Layers**: Captures long-range temporal dependencies and harmonic transitions across sequence steps.
5. **Dropout Layers**: Prevents overfitting on short musical motifs.
6. **Softmax Output**: Predicts a probability distribution over all unique vocabulary notes for the next timestep.
7. **Temperature Sampling**:
   $$\text{logits} = \frac{\log(P)}{\text{temperature}}$$
   Adjusting temperature $T$ scales entropy: lower values ($T < 1.0$) select high-probability notes (predictable), while higher values ($T > 1.0$) introduce creative randomness.

---

## 📁 Project Structure

```text
music-generation-ai/
│
├── app.py                     # Flask web server & API endpoints
├── config.py                  # Project settings & hyperparameter defaults
├── requirements.txt           # Python dependency requirements
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Version control ignore rules
├── test_pipeline.py           # Core ML pipeline unit tests
├── test_e2e.py                # Flask API integration tests
│
├── data/
│   ├── raw/                   # Raw input MIDI files
│   ├── processed/             # Preprocessed sequence pickle files
│   └── generated/             # Output generated .mid MIDI files
│
├── models/
│   └── music_lstm.h5          # Trained Keras model weights
│
├── src/
│   ├── __init__.py
│   ├── midi_processor.py      # music21 MIDI parsing & statistics
│   ├── dataset.py             # Note index mapping & sequence generator
│   ├── model.py               # Keras LSTM model architecture builder
│   ├── train.py               # Asynchronous trainer & progress callback
│   ├── generate.py            # Temperature-based next-note sampler
│   └── midi_generator.py      # Note sequence to music21 stream & MIDI export
│
├── templates/
│   └── index.html             # Bootstrap 5 Glassmorphism Web Dashboard
│
└── static/
    ├── css/
    │   └── style.css          # Custom styling & neon glow theme
    └── js/
        └── script.js          # UI controller, Chart.js & Tone.js player
```

---

## 🎬 2-Minute Quick Demo Instructions

1. **Launch App**: Run `python app.py` and open `http://127.0.0.1:5000`.
2. **Preprocess Dataset**: Click **"Preprocess Dataset"**. Watch the stats update showing extracted note count and unique pitch count.
3. **Train Model**: Click **"Start Training"**. Observe live training progress bars and real-time loss/accuracy curves updating per epoch.
4. **Generate Music**: Adjust the temperature slider to `1.0` and click **"Generate Original Music"**.
5. **Play & Download**: Click **Play** to listen to browser audio synthesis or click **Download MIDI (.mid)** to save the generated composition.

---

## 🚀 Future Improvements

- Multi-instrument track modeling (piano, violin, cello synthesis).
- Transformer / Music Transformer architecture for long-range structure.
- GAN-based music composition.
- Real-time Web MIDI input keyboard recording.
