import os
import glob
import logging
import music21
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_midi_files(directory: str) -> List[str]:
    """
    Recursively find all .mid and .midi files in the given directory.
    """
    midi_files = []
    for ext in ('*.mid', '*.midi', '*.MID', '*.MIDI'):
        midi_files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
    return sorted(list(set(midi_files)))

def extract_notes_from_file(file_path: str) -> List[str]:
    """
    Extract note and chord sequence from a single MIDI file using music21.
    Handles single Notes and Chords safely.
    """
    notes = []
    try:
        midi = music21.converter.parse(file_path)
        notes_to_parse = None

        # Check if file has instrument parts or flat structure
        try:
            parts = music21.instrument.partitionByInstrument(midi)
            if parts: # File has instrument parts
                notes_to_parse = parts.parts[0].recurse()
            else: # File has flat notes structure
                notes_to_parse = midi.flat.notes
        except Exception as e:
            logger.warning(f"Could not partition instrument for {file_path}, falling back to flat structure: {e}")
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, music21.note.Note):
                # Represent single note by pitch name with octave (e.g. 'C4', 'F#3')
                notes.append(str(element.pitch))
            elif isinstance(element, music21.chord.Chord):
                # Represent chord by dot-separated sorted pitch names (e.g. 'C4.E4.G4' or pitch numbers)
                chord_notes = '.'.join(str(n) for n in element.pitches)
                notes.append(chord_notes)

        logger.info(f"Successfully extracted {len(notes)} notes/chords from {os.path.basename(file_path)}")
    except Exception as e:
        logger.error(f"Error parsing MIDI file {file_path}: {e}")

    return notes

def parse_midi_dataset(raw_dir: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Scans raw directory, parses all MIDI files, extracts notes, and returns overall notes list + dataset statistics.
    """
    midi_files = find_midi_files(raw_dir)
    all_notes = []
    file_stats = []

    if not midi_files:
        logger.warning(f"No MIDI files found in {raw_dir}")
        return [], {
            "num_files": 0,
            "total_notes": 0,
            "unique_notes": 0,
            "pitch_range": "N/A",
            "avg_sequence_length": 0,
            "files": []
        }

    midi_pitches = []

    for fpath in midi_files:
        file_notes = extract_notes_from_file(fpath)
        all_notes.extend(file_notes)
        file_stats.append({
            "filename": os.path.basename(fpath),
            "note_count": len(file_notes)
        })

        # Calculate MIDI pitches for range detection
        for item in file_notes:
            for part in item.split('.'):
                try:
                    p = music21.pitch.Pitch(part)
                    midi_pitches.append(p.midi)
                except Exception:
                    pass

    unique_notes_count = len(set(all_notes))
    min_pitch = min(midi_pitches) if midi_pitches else 0
    max_pitch = max(midi_pitches) if midi_pitches else 0

    pitch_range_str = f"{min_pitch} - {max_pitch}" if midi_pitches else "N/A"
    avg_len = round(len(all_notes) / len(midi_files), 1) if midi_files else 0

    stats = {
        "num_files": len(midi_files),
        "total_notes": len(all_notes),
        "unique_notes": unique_notes_count,
        "pitch_range": pitch_range_str,
        "avg_sequence_length": avg_len,
        "files": file_stats
    }

    logger.info(f"Dataset summary: {stats['num_files']} files, {stats['total_notes']} total notes, {stats['unique_notes']} unique notes/chords.")
    return all_notes, stats
