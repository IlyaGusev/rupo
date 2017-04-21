# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Рекуррентная сеть для получения ударений по фонемам.

import numpy as np
import os

from typing import List, Tuple

from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import LSTM, Bidirectional, Dropout, Activation, Dense, Masking


class RNNAccentPredictor:
    phonetic_alphabet = " n̪ʃʆäʲ。ˌʰʷːːɐaɑəæbfv̪gɡxtdɛ̝̈ɬŋeɔɘɪjʝɵʂɕʐʑijkјɫlmɱnoprɾszᵻuʉɪ̯ʊɣʦʂʧʨɨɪ̯̯ɲʒûʕχѝíʌɒ‿͡ðwhɝθ"

    def __init__(self, dict_path: str, word_max_length: int=30, language: str="ru", rnn=LSTM,
                 units: int=128, dropout: float=0.2):
        self.rnn = rnn
        self.dropout = dropout  # type: float
        self.units = units  # type: int
        self.language = language  # type: str
        self.dict_path = dict_path  # type: str
        self.word_max_length = word_max_length  # type: int
        self.model = None

    def build(self) -> None:
        """
        Построение модели. 
        """
        model = Sequential()
        model.add(Masking(mask_value=0., input_shape=(self.word_max_length, len(self.phonetic_alphabet))))
        model.add(Bidirectional(self.rnn(self.units)))
        model.add(Dropout(self.dropout))
        model.add(Dense(self.word_max_length))
        model.add(Activation('softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def train(self, dir_name: str, enable_checkpoints: bool=False) -> None:
        """
        Обучение сети.
        
        :param dir_name: папка, в которую сохраняеются все весрии модели.
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
        self.model.fit(x_train, y_train, verbose=1, epochs=20, validation_data=(x_val, y_val), callbacks=callbacks)
        # Рассчёт точности на val выборке.
        accuracy = self.model.evaluate(x_val, y_val)[1]
        # Один раунд обучения на всём датасете.
        self.model.fit(x, y, verbose=1, epochs=1)
        # Сохранение модели.
        filename = "accent_{language}_{rnn}{units}_dropout{dropout}_acc{acc}.h5"
        filename = filename.format(language=self.language, rnn=self.rnn.__name__,
                                   units=self.units, dropout=self.dropout, acc=accuracy)
        self.model.save(os.path.join(dir_name, filename))

    def predict(self, words: List[str]) -> List[int]:
        """
        Предсказание ударений.
        
        :param words: слова. 
        :return: ударения.
        """
        x, y = self.__prepare_data(words, None)
        accents = [int(np.argmax(prob)) for prob in self.model.predict(x, verbose=0)]
        return accents

    def load(self, filename: str) -> None:
        self.model = load_model(filename)

    def __load_dict(self) -> Tuple[List[str], List[int]]:
        """
        Парсинг словаря.
        
        :return: фонетические слова и ударения.
        """
        x = []
        y = []
        with open(self.dict_path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                phonemes, primary, secondary = line.split("\t")
                flag = False
                for p in phonemes:
                    if p not in self.phonetic_alphabet:
                        flag = True
                if flag:
                    continue
                x.append(phonemes)
                y.append(primary)
        return x, y

    def __prepare_data(self, x: List[str], y: List[int]=None) -> Tuple[List[List[List[int]]], List[int]]:
        """
        Подготовка данных
        
        :param x: семплы.
        :param y: ответы.
        :return: очищенные семплы и овтеты.
        """
        x = [[[int(ch == ch2) for ch2 in self.phonetic_alphabet] for ch in p] for p in x]
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=np.zeros(len(self.phonetic_alphabet)),
                                   padding="post", truncating="post")
        return x, y
