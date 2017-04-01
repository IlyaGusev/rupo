from typing import List
import numpy as np
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from keras.models import Sequential


class LSTM_Container(object):
    def __init__(self, path: str):
        self.num_of_words = 60000
        self.model = Sequential()
        self.model.add(Embedding(self.num_of_words + 1, 150, mask_zero=True, batch_input_shape=(1000, None)))
        self.model.add(SpatialDropout1D(0.3))
        self.model.add(LSTM(512, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
        self.model.add(LSTM(512, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
        self.model.add(Dense(self.num_of_words + 1, activation='softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        self.model.load_weights(path)

    def get_model(self, word_indices: List[int]) -> np.array:
        if len(word_indices) == 0:
            return np.full(self.num_of_words, 1 / self.num_of_words, dtype=np.float)
        for i in range(10):
            X = np.zeros((1000, len(word_indices)))
            for ind, word_index in enumerate(word_indices):
                X[0, ind] = word_index
            return self.model.predict(X, batch_size=1000, verbose=0)[0][:-1]
