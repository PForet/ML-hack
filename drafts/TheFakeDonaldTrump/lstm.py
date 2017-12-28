import numpy as np
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from text2vec import TrumpLex

seq_length = 12
speeches_length = len(TrumpLex.parsed_text)

X, y = [], []

for i in range(speeches_length-seq_length):
    X.append(TrumpLex.get_sequence(i,seq_length))
    y.append(TrumpLex.get_sequence(i+seq_length, 1))
X, y = np.array(X), np.array(y)
y = y.reshape(y.shape[0], y.shape[2])
   
FakeTrump = Sequential()
FakeTrump.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
FakeTrump.add(Dropout(0.2))
FakeTrump.add(Dense(y.shape[1], activation='softmax'))
FakeTrump.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-lstm.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

#FakeTrump.fit(X, y, epochs=20, batch_size=128, callbacks=[checkpoint])

FakeTrump.load_weights(filepath)

current_sequence = np.array(TrumpLex.get_sequence(1,seq_length)).reshape(1,12,1001)
generated_speech = []
for i in range(100):
    new_word = FakeTrump.predict(current_sequence)
    generated_speech.append(new_word[0])
    current_sequence = np.append(current_sequence[0][1:],new_word).reshape(1,12,1001)
    print(TrumpLex.decode(new_word, remove_other=True))
    