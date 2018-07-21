import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tweepy
from tweepy import OAuthHandler
import re
import sys




if __name__ == "__main__":
	

	inFile = open("corpus.txt", "r")
	lines = [line for line in inFile.readlines()]
	raw_text = "".join(lines)
	chars = sorted(list(set(raw_text)))
	char_to_int = dict((c, i) for i, c in enumerate(chars))
	int_to_char = dict((i, c) for i, c in enumerate(chars))
	n_chars = len(raw_text)
	n_vocab = len(chars)
	seq_length = 75
	dataX = []
	dataY = []
	for i in range(0, n_chars - seq_length, 1):
		seq_in = raw_text[i:i + seq_length]
		seq_out = raw_text[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])
	n_patterns = len(dataX)
	X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
	X = X / float(n_vocab)
	# one hot encode the output variable
	y = np_utils.to_categorical(dataY)
	model = Sequential()
	model.add(LSTM(256, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
	model.add(Dropout(0.2))
	model.add(LSTM(128, input_shape=(256,)))
	model.add(Dropout(0.2))
	model.add(Dense(y.shape[1], activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	filename = "weights-improvement-20-0.4724.hdf5"
	model.load_weights(filename)
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	start = numpy.random.randint(0, len(dataX)-1)
	pattern = dataX[start]
	print(pattern)
	print("Seed:")
	print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
	print("end seed")
    # generate characters
	for i in range(140+seq_length):
	    count = 0
	    x = numpy.reshape(pattern, (1, len(pattern), 1))
	    x = x / float(n_vocab)
	    prediction = model.predict(x, verbose=0)
	    index = numpy.argmax(prediction)
	    result = int_to_char[index]
	    seq_in = [int_to_char[value] for value in pattern]
	    sys.stdout.write(result)
	    pattern.append(index)
	    pattern = pattern[1:len(pattern)]
	print("\nDone.")

