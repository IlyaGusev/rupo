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


class RNNPhonemePredictor:
    phonetic_alphabet = " n̪ʃʆäʲ。ˌʰʷːːɐaɑəæbfv̪gɡxtdɛ̝̈ɬŋeɔɘɪjʝɵʂɕʐʑijkјɫlmɱnoprɾszᵻuʉɪ̯ʊɣʦʂʧʨɨɪ̯̯ɲʒûʕχѝíʌɒ‿͡"
    # phonetic_alphabet = " n̪ʃʆäʲ。ˌʰʷːːɐaɑəæbfv̪gɡxtdɛ̝̈ɬŋeɔɘɪjʝɵʂɕʐʑijkјɫlmɱnoprɾszᵻuʉɪ̯ʊɣʦʂʧʨɨɪ̯̯ɲʒûʕχѝíʌɒ‿͡ðwhɝθ"

    def __init__(self, dict_path, word_max_length=30, language="ru"):
        if language == "ru":
            self.grapheme_alphabet = " абвгдеёжзийклмнопрстуфхцчшщьыъэюя-"
        elif language == "en":
            self.grapheme_alphabet = " abcdefghijklmnopqrstuvwxyz.'-"
        else:
            assert False
        self.dict_path = dict_path
        self.word_max_length = word_max_length
        self.model = None

    def build(self):
        input_n = len(self.grapheme_alphabet)
        output_n = len(self.phonetic_alphabet)
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(256, return_sequences=True, kernel_initializer="he_normal"),
                                     input_shape=(None, input_n)))
        self.model.add(Dropout(0.2))
        self.model.add(TimeDistributed(Dense(output_n, kernel_initializer="he_normal")))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(self.model.summary())

    def train(self, path_to_save):
        x, y = self.__load_dict()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
        x_train, y_train = self.__prepare_data(x_train, y_train)
        x_test, y_test = self.__prepare_data(x_test, y_test)
        wer_old = 1.0
        per_old = 1.0
        for i in range(40):
            self.model.fit(x_train, y_train, verbose=1, epochs=1)
            print("Epoch: " + str(i))
            wer, per = self.__validate(x_test, y_test)
            if wer > wer_old and per > per_old:
                break
            wer_old = wer
            per_old = per
            self.save(path_to_save)

    def __load_dict(self):
        x = []
        y = []
        with open(self.dict_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                g = line.split("\t")[0].strip().lower()
                p = line.split("\t")[1].strip()
                flag = False
                for ch in g:
                    if ch not in self.grapheme_alphabet:
                        flag = True
                for ch in p:
                    if ch not in self.phonetic_alphabet:
                        flag = True
                if flag:
                    continue
                x.append(g)
                y.append(p)
        return x, y

    def __prepare_data(self, x, y):
        x = [[[int(ch == ch2) for ch2 in self.grapheme_alphabet] for ch in g] for g in x]
        y = [[self.phonetic_alphabet.find(ch) for ch in p] for p in y]
        padding = np.zeros(len(self.grapheme_alphabet))
        padding[0] = 1
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=padding, padding="post", truncating="post")
        y = sequence.pad_sequences(y, maxlen=self.word_max_length, value=0, padding="post", truncating="post")
        y = y.reshape((y.shape[0], y.shape[1], 1))
        print(x.shape, y.shape)
        return x, y

    def __validate(self, x, y):
        print("Validation:")
        answer = self.model.predict(x, verbose=0)
        word_errors = 0
        phoneme_errors = 0
        for i, word in enumerate(answer):
            flag = False
            for j, ch_prob in enumerate(word):
                a = np.argmax(ch_prob)
                if y[i][j] != a:
                    phoneme_errors += 1
                    flag = True
            if flag:
                word_errors += 1

        wer = float(word_errors)/answer.shape[0]
        all_phonemes = np.ndarray.flatten(y)
        all_phonemes = all_phonemes[all_phonemes != 0]
        per = float(phoneme_errors) / len(all_phonemes)
        print("WER: " + str(wer))
        print("PER(replace only): " + str(per))
        return wer, per

    def predict(self, word):
        x = [[[int(ch == ch2) for ch2 in self.grapheme_alphabet] for ch in word]]
        padding = np.zeros(len(self.grapheme_alphabet))
        padding[0] = 1
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=padding, padding="post", truncating="post")
        word = self.model.predict(x, verbose=0)
        answer = ""
        for ch_prob in word[0]:
            i = np.argmax(ch_prob)
            answer += self.phonetic_alphabet[i]
        return answer.strip()

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

