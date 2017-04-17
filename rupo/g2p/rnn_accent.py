# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Рекуррентная сеть для получения фонем из графем.

import numpy as np
import os
import h5py
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, GRU, Dropout, Activation, TimeDistributed, Dense, Merge
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


class RNNAccentPredictor:
    phonetic_alphabet = " n̪ʃʆäʲ。ˌʰʷːːɐaɑəæbfv̪gɡxtdɛ̝̈ɬŋeɔɘɪjʝɵʂɕʐʑijkјɫlmɱnoprɾszᵻuʉɪ̯ʊɣʦʂʧʨɨɪ̯̯ɲʒûʕχѝíʌɒ‿͡ðwhɝθ"

    def __init__(self, dict_path, word_max_length=30, language="ru"):
        self.dict_path = dict_path
        self.word_max_length = word_max_length
        self.model = None

    def build(self):
        input_n = len(self.phonetic_alphabet)
        output_n = self.word_max_length
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(128), input_shape=(None, input_n)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_n))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def train(self, path_to_save):
        x, y = self.__load_dict()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        x_train, y_train = self.__prepare_data(x_train, y_train)
        x_test, y_test = self.__prepare_data(x_test, y_test)
        accuracy_error_old = 1.0
        for i in range(40):
            self.model.fit(x_train, y_train, verbose=1, epochs=1)
            print("Epoch: " + str(i))
            accuracy_error = self.__validate(x_test, y_test)
            if accuracy_error > accuracy_error_old:
                break
            accuracy_error_old = accuracy_error
            self.save(path_to_save)

    def __load_dict(self):
        x = []
        y = []
        with open(self.dict_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                p = line.split("\t")[0].strip()
                f = line.split("\t")[1].strip()
                flag = False
                for ch in p:
                    if ch not in self.phonetic_alphabet:
                        flag = True
                if flag:
                    continue
                x.append(p)
                y.append(f)
        return x, y

    def __prepare_data(self, x, y):
        x = [[[int(ch == ch2) for ch2 in self.phonetic_alphabet] for ch in p] for p in x]
        padding = np.zeros(len(self.phonetic_alphabet))
        padding[0] = 1
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=padding, padding="post", truncating="post")
        return x, y

    def __validate(self, x, y):
        print("Validation:")
        answer = self.model.predict(x, verbose=0)
        errors = 0
        for i, prob in enumerate(answer):
            a = np.argmax(prob)
            if int(a) != int(y[i]):
                errors += 1

        print(errors)
        acc_error = float(errors)/len(y)
        print("Accuracy error: " + str(acc_error))
        return acc_error

    def predict(self, word):
        x = [[[int(ch == ch2) for ch2 in self.phonetic_alphabet] for ch in word]]
        padding = np.zeros(len(self.phonetic_alphabet))
        padding[0] = 1
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=padding, padding="post", truncating="post")
        accent = np.argmax(self.model.predict(x, verbose=0)[0])
        return accent

    def save(self, weights_path):
        if os.path.exists(weights_path):
            os.remove(weights_path)
        file = h5py.File(weights_path, 'w')
        weight = self.model.get_weights()
        for i in range(len(weight)):
            file.create_dataset('weight' + str(i), data=weight[i])
        file.close()

    def load(self, weights_path):
        file = h5py.File(weights_path, 'r')
        weight = []
        for i in range(len(file.keys())):
            weight.append(file['weight' + str(i)][:])
        self.model.set_weights(weight)
