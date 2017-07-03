# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Рекуррентная сеть для получения фонем из графем.

import numpy as np
import os
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.preprocessing import sequence
from keras.layers import LSTM, Bidirectional, Dropout, TimeDistributed, Dense, Input, Embedding
from keras.layers.merge import concatenate
from rupo.settings import RU_GRAPHEME_SET, EN_GRAPHEME_SET
from rupo.g2p.phonemes import Phonemes


class RNNG2PModel:
    phonetic_alphabet = "".join(Phonemes.get_all())

    def __init__(self, dict_path: str=None, word_max_length: int=30, language: str = "ru", rnn=LSTM,
                 units1: int=256, units2: int=256, dropout: float = 0.2, batch_size=2048, emb_dimension=30):
        self.rnn = rnn
        self.dropout = dropout  # type: float
        self.units1 = units1  # type: int
        self.units2 = units2  # type: int
        self.language = language  # type: str
        if language == "ru":
            self.grapheme_alphabet = RU_GRAPHEME_SET
        elif language == "en":
            self.grapheme_alphabet = EN_GRAPHEME_SET
        else:
            assert False
        self.dict_path = dict_path  # type: str
        self.word_max_length = word_max_length  # type: int
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.model = None

    def build(self) -> None:
        """
        Построение модели.
        """
        inp = Input(shape=(None,))

        emb = Embedding(len(self.grapheme_alphabet), self.emb_dimension)(inp)
        encoded = Bidirectional(self.rnn(self.units1//2, return_sequences=True, recurrent_dropout=self.dropout))(emb)
        encoded = Dropout(self.dropout)(encoded)
        encoded = TimeDistributed(Dense(self.units2, activation="relu"))(encoded)
        decoded = Bidirectional(self.rnn(self.units2//2, return_sequences=True, recurrent_dropout=self.dropout))(encoded)
        decoded = Dropout(self.dropout)(decoded)
        predictions = TimeDistributed(Dense(len(self.phonetic_alphabet), activation="softmax"))(decoded)

        model = Model(inputs=inp, outputs=predictions)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())
        self.model = model

    def train(self, dir_name: str, enable_checkpoints: bool = False, checkpoint: str = None) -> None:
        """
        Обучение модели.

        :param dir_name: папка с версиями модели.
        :param enable_checkpoints: использовать ли чекпоинты.
        :param checkpoint: загрузка чекпоинта.
        """
        # Подготовка данных
        x, y = self.load_dict()
        x, y = self.prepare_data(x, y)
        # Деление на выборки.
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
        x_test, x_val, y_test, y_val = train_test_split(x_val, y_val, test_size=0.5, random_state=42)
        # Основные раунды обучения.
        callbacks = [EarlyStopping(monitor='val_acc', patience=2)]  # type: List[Callback]
        if enable_checkpoints:
            checkpoint_name = os.path.join(dir_name, "{epoch:02d}-{val_loss:.2f}.hdf5")
            callbacks.append(ModelCheckpoint(checkpoint_name, monitor='val_loss'))
        if checkpoint is not None:
            self.load(checkpoint)
        self.model.fit(x_train, y_train, verbose=1, epochs=60, batch_size=self.batch_size,
                       validation_data=(x_val, y_val), callbacks=callbacks)
        # Рассчёт точности и word error rate на test выборке.
        accuracy = self.model.evaluate(x_test, y_test)[1]
        wer = self.evaluate_wer(x_test, y_test)[0]
        # Один раунд обучения на всём датасете.
        self.model.fit(x, y, verbose=1, epochs=1, batch_size=self.batch_size)
        # Сохранение модели.
        filename = "g2p_{language}_maxlen{maxlen}_B{rnn}{units1}_B{rnn}{units2}_dropout{dropout}_acc{acc}_wer{wer}.h5"
        filename = filename.format(language=self.language, rnn=self.rnn.__name__,
                                   units1=self.units1, units2=self.units2, dropout=self.dropout,
                                   acc=int(accuracy * 1000), wer=int(wer * 1000), maxlen=self.word_max_length)
        self.model.save(os.path.join(dir_name, filename))

    def predict(self, words: List[str]) -> List[str]:
        """
        Трансляция в фонемы.

        :param words: графические слова.
        :return: фонетические слова.
        """
        x = self.prepare_data(words, None)[0]
        y = self.model.predict(x, verbose=0, batch_size=self.batch_size)
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

    def evaluate_wer(self, x: np.array, y: np.array) -> Tuple[float, float]:
        """
        Считаем word error rate - количество слов, в которых была допущена 
        хоть одна ошибка при транскрибировании.

        :param x: данные.
        :param y: ответы
        :return: метрики.
        """
        print("Validation:")
        answer = self.model.predict(x, verbose=0, batch_size=self.batch_size)
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

        wer = float(word_errors) / answer.shape[0]
        all_phonemes = np.ndarray.flatten(y)
        per = float(phoneme_errors) / len(all_phonemes[all_phonemes != 0])
        print("WER: " + str(wer))
        print("PER(replace only with zeros): " + str(float(phoneme_errors) / len(all_phonemes)))
        print("PER(replace only): " + str(per))
        return wer, per

    def load_dict(self) -> Tuple[List[str], List[str]]:
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

    def prepare_data(self, x: List[str], y: List[str] = None) -> Tuple[np.array, np.array]:
        """
        Подготовка данных.

        :param x: графические слова.
        :param y: фонетические слова.
        :return: данные в числовом виде.
        """
        x = [[self.grapheme_alphabet.find(ch) for ch in g] for g in x]
        x = sequence.pad_sequences(x, maxlen=self.word_max_length, value=0)
        if y is not None:
            y = [[self.phonetic_alphabet.find(ch) for ch in p] for p in y]
            y = sequence.pad_sequences(y, maxlen=self.word_max_length, value=0)
            y = y.reshape((y.shape[0], y.shape[1], 1))
        return x, y