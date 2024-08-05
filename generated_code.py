# Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.callbacks import ModelCheckpoint

np.random.seed(42) # For reproducibility

# Loading Data

def load_data(file_names):
    frames = []
    for file in file_names:
        df = pd.read_csv(file)
        frames.append(df)
    return pd.concat(frames)

# Preprocessing RNA Sequence

def clean_sequences(sequence_data):
    sequence_data = sequence_data.str.upper()
    valid_chars = {'A', 'C', 'G', 'T', 'U'}
    return sequence_data[~sequence_data.apply(lambda seq: any(char not in valid_chars for char in seq))]

def one_hot_encode(sequences, max_len):
    mappings = {'A': [1, 0, 0, 0, 0],'C': [0, 1, 0, 0, 0], 'G': [0, 0, 1, 0, 0],'T': [0, 0, 0, 1, 0], 'U': [0, 0, 0, 0, 1]}
    encoded = sequences.apply(lambda seq: [mappings[char] for char in seq] + [[0]*5]*(max_len-len(seq)))
    return np.array(encoded.tolist())

# Encoding Labels

def encode_labels(labels):
    return labels.map({ 'experimental': 1, 'control': 0 })

# Building the Model

def build_model(max_len, embedding_dims=32):
    model = Sequential()
    model.add(Embedding(input_dim=5, output_dim=embedding_dims, input_length=max_len, input_shape=(max_len,)))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Training and Evaluating the Model

def train_and_evaluate_model(model, x_train, y_train, x_val, y_val, epochs=20, batch_size=32):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('model_best_weights.h5', save_best_only=True, monitor='val_loss', mode='min')
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[checkpoint])
    score = model.evaluate(x_val, y_val)
    return score

files = ['file1.csv', 'file2.csv', 'file3.csv', 'file4.csv', 'file5.csv', 'file6.csv', 'file7.csv', 'file8.csv', 'file9.csv', 'file10.csv', 
         'file11.csv', 'file12.csv', 'file13.csv', 'file14.csv', 'file15.csv', 'file16.csv', 'file17.csv', 'file18.csv', 'file19.csv', 'file20.csv']

data = load_data(files)
data.sequence = clean_sequences(data.sequence)
SEQUENCE_MAX_LEN = data.sequence.apply(len).max()
data.sequence = one_hot_encode(data.sequence, SEQUENCE_MAX_LEN)
data.label = encode_labels(data.label)

x = data.sequence
y = data.label

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)

model = build_model(SEQUENCE_MAX_LEN)
score = train_and_evaluate_model(model, x_train, y_train, x_val, y_val)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
