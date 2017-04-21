# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Рекуррентная сеть для получения фонем из графем.

import numpy as np
import os
from typing import  List, Tuple

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.layers import LSTM, Bidirectional, Dropout, Activation, TimeDistributed, Dense


class RNNPhonemePredictor:
    phonetic_alphabet = " n̪ʃʆäʲ。ˌʰʷːːɐaɑəæbfv̪gɡxtdɛ̝̈ɬŋeɔɘɪjʝɵʂɕʐʑijkјɫlmɱnoprɾszᵻuʉɪ̯ʊɣʦʂʧʨɨɪ̯̯ɲʒûʕχѝíʌɒ‿͡ðwhɝθ"

    def __init__(self, dict_path: str, word_max_length: int=30, language: str="ru", rnn=LSTM,
                 units1: int=256, units2: int=128, dropout: float=0.2):
        self.rnn = rnn
        self.dropout = dropout  # type: float
        self.units1 = units1  # type: int
        self.units2 = units2  # type: int
        self.language = language  # type: str
        if language == "ru":
            self.grapheme_alphabet = " абвгдеёжзийклмнопрстуфхцчшщьыъэюя-"
        elif language == "en":
            self.grapheme_alphabet = " abcdefghijklmnopqrstuvwxyz.'-"
        else:
            assert False
        self.dict_path = dict_path  # type: str
        self.word_max_length = word_max_length  # type: int
        self.model = None

    def build(self) -> None:
        """
        Построение модели.
        """
        input_n = len(self.grapheme_alphabet)
        model = Sequential()
        model.add(Bidirectional(self.rnn(self.units1, return_sequences=True), input_shape=(None, input_n)))
        model.add(self.rnn(self.units2, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(TimeDistributed(Dense(len(self.phonetic_alphabet))))
        model.add(Activation('softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def train(self, dir_name: str, enable_checkpoints: bool=False) -> None:
        """
        Обучение модели.
        
        :param dir_name: папка с версиями модели.
        :param enable_checkpoints: использовать ли чекпоинты.
        """
        # Подготовка данных
        x, y = self.__load_dict()
        x, y = self.__prepare_data(x, y)
        # Деление на выборки.
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)
        # Основные раунды обучения.
        callbacks = [EarlyStopping(monitor='val_loss', patience=2)]  # type: List[Callback]
        if enable_checkpoints:
            checkpoint_name = os.path.join(dir_name, "{epoch:02d}-{val_loss:.2f}.hdf5")
            callbacks.append(ModelCheckpoint(checkpoint_name, monitor='val_loss'))
        self.model.fit(x_train, y_train, verbose=1, epochs=40, validation_data=(x_val, y_val), callbacks=callbacks)
        # Рассчёт точности на val выборке, расчёт word error rate на всём датасете.
        accuracy = self.model.evaluate(x_val, y_val)[1]
        wer = self.__evaluate_wer(x, y)[0]
        # Один раунд обучения на всём датасете.
        self.model.fit(x, y, verbose=1, epochs=1)
        # Сохранение модели.
        filename = "g2p_{language}_{rnn}{units1}_{rnn}{units2}_dropout{dropout}_acc{acc}_wer{wer}.h5"
        filename = filename.format(language=self.language, rnn=self.rnn.__name__,
                                   units1=self.units1, units2=self.units2, dropout=self.dropout,
                                   acc=int(accuracy*100), wer=int(wer*100))
        self.model.save(os.path.join(dir_name, filename))

    def predict(self, words: List[str]) -> List[str]:
        """
        Трансляция в фонемы.
        
        :param words: графические слова.
        :return: фонетические слова.
        """
        x = [[[int(ch == ch2) for ch2 in self.grapheme_alphabet] for ch in word] for word in words]
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=np.zeros(len(self.grapheme_alphabet)),
                                   padding="post", truncating="post")
        y = self.model.predict(x, verbose=0)
        answers = []
        for word in y:
            answer = ""
            for ch_prob in word:
                i = int(np.argmax(ch_prob))
                answer += self.phonetic_alphabet[i]
            answers.append(answer.strip())
        return answers

    def load(self, filename: str) -> None:
        self.model = load_model(filename)

    def __load_dict(self) -> Tuple[List[str], List[str]]:
        """
        Загрузка из словаря g2p.
        
        :return: графические и фонетические слова.
        """
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

    def __prepare_data(self, x: List[str], y: List[str]) -> Tuple[np.array, np.array]:
        """
        Подготовка данных.
        
        :param x: графические слова.
        :param y: фонетические слова.
        :return: данные в числовом виде.
        """
        x = [[[int(ch == ch2) for ch2 in self.grapheme_alphabet] for ch in g] for g in x]
        y = [[self.phonetic_alphabet.find(ch) for ch in p] for p in y]
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=np.zeros(len(self.grapheme_alphabet)),
                                   padding="post", truncating="post")
        y = sequence.pad_sequences(y, maxlen=self.word_max_length, value=0, padding="post", truncating="post")
        y = y.reshape((y.shape[0], y.shape[1], 1))
        return x, y

    def __evaluate_wer(self, x: np.array, y: np.array) -> Tuple[float, float]:
        """
        Считаем word error rate - количество слов, в которых была допущена 
        хоть одна ошибка при транскрибировании.
        
        :param x: данные.
        :param y: ответы
        :return: метрики.
        """
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
