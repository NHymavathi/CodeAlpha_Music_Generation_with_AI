import os
import music21

def create_sample_midis(raw_dir: str):
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. Bach Invention Style Sample
    s1 = music21.stream.Stream()
    s1.append(music21.instrument.Piano())
    s1.append(music21.tempo.MetronomeMark(number=110))
    # Notes sequence (Bach-like C Major motif)
    notes1 = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'G4', 'E4', 'C4', 
              'D4', 'E4', 'F4', 'D4', 'B3', 'C4', 'G3', 'C4', 'E4', 'G4', 'C5',
              'C4.E4.G4', 'F4.A4.C5', 'G4.B4.D5', 'C4.E4.G4', 'C4']
    offset = 0.0
    for n in notes1:
        if '.' in n:
            chord_objs = [music21.note.Note(p) for p in n.split('.')]
            c = music21.chord.Chord(chord_objs)
            c.offset = offset
            c.quarterLength = 1.0 if 'C4.E4.G4' in n else 0.5
            s1.append(c)
        else:
            note_obj = music21.note.Note(n)
            note_obj.offset = offset
            note_obj.quarterLength = 0.5
            s1.append(note_obj)
        offset += 0.5
    s1.write('midi', fp=os.path.join(raw_dir, 'sample_bach_invention.mid'))

    # 2. Beethoven Sonata Style Sample
    s2 = music21.stream.Stream()
    s2.append(music21.instrument.Piano())
    s2.append(music21.tempo.MetronomeMark(number=90))
    notes2 = ['E4', 'E4', 'F4', 'G4', 'G4', 'F4', 'E4', 'D4', 'C4', 'C4', 'D4', 'E4',
              'E4', 'D4', 'D4', 'E4', 'E4', 'F4', 'G4', 'G4', 'F4', 'E4', 'D4',
              'C4', 'C4', 'D4', 'E4', 'D4', 'C4', 'C4',
              'C4.E4.G4', 'G3.B3.D4', 'A3.C4.E4', 'F3.A3.C4']
    offset = 0.0
    for n in notes2:
        if '.' in n:
            c = music21.chord.Chord([music21.note.Note(p) for p in n.split('.')])
            c.offset = offset
            c.quarterLength = 1.0
            s2.append(c)
        else:
            note_obj = music21.note.Note(n)
            note_obj.offset = offset
            note_obj.quarterLength = 0.5
            s2.append(note_obj)
        offset += 0.5
    s2.write('midi', fp=os.path.join(raw_dir, 'sample_beethoven_theme.mid'))

    # 3. Mozart Minuet Style Sample
    s3 = music21.stream.Stream()
    s3.append(music21.instrument.Piano())
    s3.append(music21.tempo.MetronomeMark(number=120))
    notes3 = ['G4', 'D4', 'E4', 'F#4', 'G4', 'G4', 'A4', 'D4', 'B4', 'G4', 'C5', 'B4', 'A4',
              'G4', 'F#4', 'G4', 'A4', 'D4', 'F#4', 'A4', 'C5', 'B4', 'G4', 'A4',
              'G4.B4.D5', 'D4.F#4.A4', 'G4.B4.D5']
    offset = 0.0
    for n in notes3:
        if '.' in n:
            c = music21.chord.Chord([music21.note.Note(p) for p in n.split('.')])
            c.offset = offset
            c.quarterLength = 1.0
            s3.append(c)
        else:
            note_obj = music21.note.Note(n)
            note_obj.offset = offset
            note_obj.quarterLength = 0.5
            s3.append(note_obj)
        offset += 0.5
    s3.write('midi', fp=os.path.join(raw_dir, 'sample_mozart_minuet.mid'))

    # 4. Chopin Prelude Style Sample
    s4 = music21.stream.Stream()
    s4.append(music21.instrument.Piano())
    s4.append(music21.tempo.MetronomeMark(number=80))
    notes4 = ['A4', 'C5', 'E5', 'A5', 'G#5', 'E5', 'B4', 'E5', 'G5', 'F#5', 'D5', 'A4',
              'D5', 'F5', 'E5', 'C5', 'G4', 'C5', 'E5', 'A4.C5.E5', 'F4.A4.C5', 'E4.G#4.B4', 'A4.C5.E5']
    offset = 0.0
    for n in notes4:
        if '.' in n:
            c = music21.chord.Chord([music21.note.Note(p) for p in n.split('.')])
            c.offset = offset
            c.quarterLength = 1.0
            s4.append(c)
        else:
            note_obj = music21.note.Note(n)
            note_obj.offset = offset
            note_obj.quarterLength = 0.5
            s4.append(note_obj)
        offset += 0.5
    s4.write('midi', fp=os.path.join(raw_dir, 'sample_chopin_prelude.mid'))

    print(f"Successfully generated 4 sample classical MIDI files in {raw_dir}")

if __name__ == '__main__':
    create_sample_midis('data/raw')
