import os
import logging
import music21
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_midi_from_notes(
    predicted_notes: List[str],
    output_filepath: str,
    note_duration: float = 0.5,
    tempo_bpm: int = 120
) -> str:
    """
    Converts a sequence of note/chord string representations into a valid playable MIDI file.
    """
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    offset = 0.0
    output_notes = []

    # Add instrument and tempo
    output_notes.append(music21.instrument.Piano())
    output_notes.append(music21.tempo.MetronomeMark(number=tempo_bpm))

    for pattern in predicted_notes:
        # Check if pattern is a chord (contains dot '.')
        if '.' in pattern:
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                try:
                    new_note = music21.note.Note(current_note)
                    new_note.storedInstrument = music21.instrument.Piano()
                    chord_notes.append(new_note)
                except Exception as e:
                    logger.warning(f"Error parsing pitch '{current_note}' in chord pattern '{pattern}': {e}")

            if chord_notes:
                new_chord = music21.chord.Chord(chord_notes)
                new_chord.offset = offset
                output_notes.append(new_chord)
        else:
            # Single note pattern
            try:
                new_note = music21.note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = music21.instrument.Piano()
                output_notes.append(new_note)
            except Exception as e:
                logger.warning(f"Error parsing single pitch '{pattern}': {e}")

        offset += note_duration

    midi_stream = music21.stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filepath)

    if os.path.exists(output_filepath) and os.path.getsize(output_filepath) > 0:
        logger.info(f"Successfully generated MIDI file at {output_filepath} ({os.path.getsize(output_filepath)} bytes)")
    else:
        logger.error(f"Failed to generate valid MIDI file at {output_filepath}")

    return output_filepath
