# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Рекуррентная сеть для получения ударений по фонемам.

import numpy as np
import os
import logging

from typing import List, Tuple

from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import LSTM, Bidirectional, Dropout, Dense, TimeDistributed, Input, Embedding

from rupo.settings import RU_GRAPHEME_SET, RU_GRAPHEME_STRESS_PATH, DATA_DIR


class RNNGraphemeStressModel:
    def __init__(self, dict_path: str=None, word_max_length: int=30, language: str="ru", rnn=LSTM,
                 units: int=64, dropout: float=0.2, batch_size=2048, emb_dimension=30):
        self.rnn = rnn
        self.dropout = dropout  # type: float
        self.units = units  # type: int
        self.language = language  # type: str
        self.dict_path = dict_path  # type: str
        self.word_max_length = word_max_length  # type: int
        self.batch_size = batch_size
        self.emb_dimension = emb_dimension
        self.model = None
        if language == "ru":
            self.grapheme_set = RU_GRAPHEME_SET

    def build(self) -> None:
        """
        Построение модели. 
        """
        inp = Input(shape=(None,))

        emb = Embedding(len(self.grapheme_set), self.emb_dimension)(inp)
        encoded = Bidirectional(self.rnn(self.units, return_sequences=True, recurrent_dropout=self.dropout))(emb)
        encoded = Dropout(self.dropout)(encoded)
        decoded = TimeDistributed(Dense(self.units, activation="relu"))(encoded)
        predictions = TimeDistributed(Dense(3, activation="softmax"))(decoded)

        model = Model(inputs=inp, outputs=predictions)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def train(self, dir_name: str, enable_checkpoints: bool = False) -> None:
        """
        Обучение сети.

        :param dir_name: папка, в которую сохраняеются все весрии модели.
        :param enable_checkpoints: использовать ли чекпоинты.
        """
        # Подготовка данных
        x, y = self.__load_dict()
        x, y = self.__prepare_data(x, y)
        # Деление на выборки.
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size=0.5, random_state=42)
        # Основные раунды обучения.
        callbacks = [EarlyStopping(monitor='val_acc', patience=3)]  # type: List[Callback]
        if enable_checkpoints:
            checkpoint_name = os.path.join(dir_name, "checkpoint.hdf5")
            callbacks.append(ModelCheckpoint(checkpoint_name, monitor='val_loss'))
        self.model.fit(x_train, y_train, verbose=1, epochs=200, validation_data=(x_val, y_val),
                       callbacks=callbacks, batch_size=self.batch_size)
        # Рассчёт точности на test выборке.
        accuracy = self.model.evaluate(x_test, y_test)[1]
        # Расчёт WER на test выборке.
        wer = self.__evaluate_wer(x_test, y_test)[0]
        # Один раунд обучения на всём датасете.
        self.model.fit(x, y, verbose=1, epochs=1, batch_size=self.batch_size)
        # Сохранение модели.
        filename = "stress_{language}_{rnn}{units}_dropout{dropout}_acc{acc}_wer{wer}.h5"
        filename = filename.format(language=self.language, rnn=self.rnn.__name__,
                                   units=self.units, dropout=self.dropout, acc=int(accuracy * 100),
                                   wer=int(wer * 100))
        self.model.save(os.path.join(dir_name, filename))

    def predict(self, words: List[str]) -> List[List[int]]:
        """
        Предсказание ударений.

        :param words: слова. 
        :return: ударения.
        """
        x, y = self.__prepare_data(words, None)
        y = self.model.predict(x, verbose=0, batch_size=self.batch_size)
        answers = []
        for word in y:
            answer = []
            for ch_prob in word:
                i = int(np.argmax(ch_prob))
                answer.append(i)
            answers.append(answer)
        return answers

    def load(self, filename: str) -> None:
        self.model = load_model(filename)

    def __load_dict(self) -> Tuple[List[str], np.array]:
        """
        Парсинг словаря.

        :return: графические слова и ударения.
        """
        x = []
        y = []
        skipped = 0
        with open(self.dict_path, "r", encoding='utf-8') as f:
            for line in f:
                graphemes, primary, secondary = line.split("\t")
                graphemes = graphemes.lower()
                primary = [int(i) for i in primary.split(",") if i != '']
                secondary = [int(i) for i in secondary.strip().split(",") if i != '']
                if len(graphemes) > self.word_max_length:
                    skipped += 1
                    continue
                flag = False
                for g in graphemes:
                    if g not in self.grapheme_set:
                        flag = True
                if flag:
                    continue
                stress_mask = np.zeros(self.word_max_length)
                for stress in secondary:
                    stress_mask[stress] = 2
                for stress in primary:
                    stress_mask[stress] = 1
                x.append(graphemes)
                y.append(stress_mask)
        y = np.array(y)
        logging.debug("Skipped: " + str(skipped))
        return x, y

    def __prepare_data(self, x: List[str], y: np.array = None) -> Tuple[np.array, List[int]]:
        """
        Подготовка данных

        :param x: семплы.
        :param y: ответы.
        :return: очищенные семплы и овтеты.
        """
        x = [[self.grapheme_set.find(ch) if ch in self.grapheme_set else 0 for ch in p] for p in x]
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, padding='post', truncating='post')
        if y is not None:
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
        stress_errors = 0
        for i, word in enumerate(answer):
            flag = False
            for j, ch_prob in enumerate(word):
                a = np.argmax(ch_prob)
                if y[i][j] != a:
                    stress_errors += 1
                    flag = True
            if flag:
                word_errors += 1

        wer = float(word_errors) / answer.shape[0]
        all_phonemes = np.ndarray.flatten(y)
        all_phonemes = all_phonemes[all_phonemes != 0]
        per = float(stress_errors) / len(all_phonemes)
        print("WER: " + str(wer))
        print("PER: " + str(per))
        return wer, per
