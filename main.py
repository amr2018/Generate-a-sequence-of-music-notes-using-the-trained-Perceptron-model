
import mido
from sklearn.linear_model import Perceptron
import joblib
import os
import random



# Function to read MIDI files and extract notes
def get_notes(midi_folder):
    notes = []

    try:
        for file in os.listdir(midi_folder):
            midi_path = f'{midi_folder}/{file}'
            for msg in mido.MidiFile(midi_path):
                if not msg.is_meta and msg.type == 'note_on':
                    notes.append(msg.note)

    except:
        pass

    
    
    return notes


X = []
y = []
sec_note_len = 50

def process_notes():
    notes = get_notes('dataset')
    notes = list(set(notes))
    for i in range(0, len(notes) - sec_note_len, 1):
        sec_in = notes[i : i + sec_note_len]
        sec_out = notes[i + sec_note_len]
        X.append([c for c in sec_in])
        y.append(sec_out)


process_notes()


model = Perceptron()


if os.path.exists('my_model.pkl'):
    model = joblib.load('my_model.pkl')
else:
    print('Training ...')
    model.fit(X, y)
    joblib.dump(model, 'my_model.pkl')
    model = joblib.load('my_model.pkl')


def generate_music(model, start_notes, num_notes, unique_notes):
    """
    Generate a sequence of music notes using the trained Perceptron model.

    Parameters:
    model: Trained Perceptron model.
    start_notes: Initial sequence of notes to start the prediction.
    num_notes: Number of notes to generate.
    unique_notes: Set of unique notes to ensure generated notes are valid.

    Returns:
    List of generated music notes.
    """
    generated_notes = start_notes[:]
    for _ in range(num_notes):
        # Prepare the input for the model
        input_notes = generated_notes[-sec_note_len:]
        input_notes_array = [input_notes]

        # Predict the next note
        predicted_note = int(model.predict(input_notes_array)[0])
        
        

        # Ensure the predicted note is within the range of unique notes
        # if predicted_note not in unique_notes:
        #     predicted_note = random.choice(list(unique_notes))

        # Append the predicted note to the generated notes
        generated_notes.append(predicted_note)

    return generated_notes

# Usage example
unique_notes = set(get_notes('dataset'))  # Get the set of unique notes from the dataset
index = 20
start_notes = get_notes('dataset')[index:sec_note_len+index]  # Starting sequence of notes

generated_sequence = generate_music(model, start_notes, 100, unique_notes)  # Generate 50 notes
#print(generated_sequence)
# Convert the sequence of notes into a MIDI file
output_midi = mido.MidiFile()
track = mido.MidiTrack()
output_midi.tracks.append(track)

for note in generated_sequence:
    # Add note_on and note_off messages for each note
    track.append(mido.Message('note_on', note=note, velocity=64, time=120))
    track.append(mido.Message('note_off', note=note, velocity=64, time=120))

# Save the generated MIDI file
output_midi.save('generated_music.mid')

