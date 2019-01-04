# -*- coding: utf-8 -*-
# Авторы: Анастасьев Даниил, Гусев Илья
# Описание: Модуль рекуррентой сети для генерации языковой модели.

import os
from itertools import islice
from typing import List, Tuple

import numpy as np
from keras.layers import Input, Embedding, Dense, LSTM, SpatialDropout1D, BatchNormalization, \
    Activation, concatenate, TimeDistributed, Bidirectional, Dropout
from keras.models import Model, load_model

from rupo.generate.language_model.model_container import ModelContainer
from rupo.generate.language_model.batch_generator import BatchGenerator, CHAR_SET
from rupo.generate.prepare.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.prepare.loader import CorporaInformationLoader
from rupo.generate.prepare.word_form_vocabulary import WordFormVocabulary
from rupo.settings import GENERATOR_LSTM_MODEL_PATH, GENERATOR_WORD_FORM_VOCAB_PATH, GENERATOR_GRAM_VECTORS


class LSTMGenerator:
    """
    Языковая модель на основе двухуровневой LSTM RNN.
    """
    def __init__(self, embedding_size: int=30000, external_batch_size: int=10000, nn_batch_size: int=768,
                 sentence_maxlen: int=10, lstm_units=368, embeddings_dimension: int=150, 
                 grammeme_dense_units: Tuple[int]=(35, 15), dense_units: int=256, softmax_size: int=60000,
                 dropout: float=0.2, recalculate_softmax=False, max_word_len: int=30, char_embeddings_dimension: int=20,
                 char_lstm_output_dim: int=64, input_dense_size: int=128):
        """
        :param embedding_size: размер входного слоя (=размер словаря)
        :param softmax_size: размер выхода softmax-слоя (=размер итогового набора вероятностей)
        :param external_batch_size: размер набора семплов для BatchGenerator'а.
        :param nn_batch_size: размер набора семплов для обучения.
        :param sentence_maxlen: маскимальная длина куска предложения.
        """
        self.embedding_size = embedding_size  # type: int
        self.softmax_size = softmax_size  # type: int
        self.external_batch_size = external_batch_size  # type: int
        self.nn_batch_size = nn_batch_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_form_vocabulary = None  # type: WordFormVocabulary
        self.grammeme_vectorizer = None  # type: GrammemeVectorizer
        self.lstm_units = lstm_units  # type: int
        self.embeddings_dimension = embeddings_dimension  # type: int
        self.grammeme_dense_units = grammeme_dense_units  # type: List[int]
        self.dense_units = dense_units  # type: int
        self.dropout = dropout  # type: float
        self.input_dense_size = input_dense_size  # type: int
        self.max_word_len = max_word_len  # type: int
        self.char_embeddings_dimension = char_embeddings_dimension  # type: int
        self.char_lstm_output_dim = char_lstm_output_dim  # type: int
        self.model = None  # type: Model
        self.recalculate_softmax = recalculate_softmax  # type: bool

    def prepare(self, filenames: List[str]=list(),
                word_form_vocab_dump_path: str=GENERATOR_WORD_FORM_VOCAB_PATH,
                gram_dump_path: str=GENERATOR_GRAM_VECTORS) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря словоформ по корпусу.

        :param filenames: имена файлов с морфоразметкой.
        :param word_form_vocab_dump_path: путь к дампу словаря словоформ.
        :param gram_dump_path: путь к векторам грамматических значений.
        """
        self.grammeme_vectorizer = GrammemeVectorizer()
        if os.path.isfile(gram_dump_path):
            self.grammeme_vectorizer.load(gram_dump_path)
        self.word_form_vocabulary = WordFormVocabulary()
        if os.path.isfile(word_form_vocab_dump_path):
            self.word_form_vocabulary.load(word_form_vocab_dump_path)
        if self.grammeme_vectorizer.is_empty() or self.word_form_vocabulary.is_empty():
            loader = CorporaInformationLoader(gram_dump_path, word_form_vocab_dump_path)
            self.word_form_vocabulary, self.grammeme_vectorizer = loader.parse_corpora(filenames)
            self.grammeme_vectorizer.save()
            self.word_form_vocabulary.save()
        if self.recalculate_softmax:
            self.softmax_size = self.word_form_vocabulary.get_softmax_size_by_lemma_size(self.embedding_size)
            print("Recalculated softmax: ", self.softmax_size)

    def save(self, model_filename: str):
        self.model.save(model_filename)

    def load(self, model_filename: str) -> None:
        self.model = load_model(model_filename)

    def build(self, use_chars=False):
        """
        Описание модели.
        """
        inputs = []
        embedding_parts = []

        # Вход лемм
        lemmas = Input(shape=(None,), name='lemmas')
        inputs.append(lemmas)
        lemmas_embedding = Embedding(self.embedding_size + 1,
                                     self.embeddings_dimension, name='embeddings')(lemmas)
        lemmas_embedding = SpatialDropout1D(self.dropout)(lemmas_embedding)
        embedding_parts.append(lemmas_embedding)

        # Вход граммем
        grammemes_input = Input(shape=(None, self.grammeme_vectorizer.grammemes_count()), name='grammemes')
        inputs.append(grammemes_input)
        grammemes_layer = grammemes_input
        for grammeme_dense_layer_units in self.grammeme_dense_units:
            grammemes_layer = Dense(grammeme_dense_layer_units, activation='relu')(grammemes_layer)
            embedding_parts.append(grammemes_layer)

        # Вход символов
        if use_chars:
            chars = Input(shape=(None, self.max_word_len), name='chars')
            inputs.append(chars)
            chars_embedding = Embedding(len(CHAR_SET) + 1, self.char_embeddings_dimension, name='char_embeddings')(chars)
            chars_lstm = TimeDistributed(Bidirectional(
                LSTM(self.char_lstm_output_dim // 2, dropout=self.dropout, recurrent_dropout=self.dropout,
                     return_sequences=False, name='CharLSTM')))(chars_embedding)
            embedding_parts.append(chars_lstm)

        layer = concatenate(embedding_parts, name="LSTM_input")
        layer = Dense(self.input_dense_size)(layer)
        layer = LSTM(self.lstm_units, dropout=self.dropout, recurrent_dropout=self.dropout,
                     return_sequences=True, name='LSTM_1')(layer)
        layer = LSTM(self.lstm_units, dropout=self.dropout, recurrent_dropout=self.dropout,
                     return_sequences=False, name='LSTM_2')(layer)

        layer = Dense(self.dense_units)(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)
        layer = Dropout(self.dropout)(layer)

        output = Dense(self.softmax_size + 1, activation='softmax')(layer)

        self.model = Model(inputs=inputs, outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        print(self.model.summary())

    @staticmethod
    def __get_validation_data(batch_generator, size):
        """
        Берет первые size батчей из batch_generator для валидационной выборки
        """
        lemmas_list, grammemes_list, chars_list, y_list = [], [], [], []
        for lemmas, grammemes, chars, y in islice(batch_generator, size):
            lemmas_list.append(lemmas)
            grammemes_list.append(grammemes)
            chars_list.append(chars)
            y_list.append(y)
        return np.vstack(lemmas_list), np.vstack(grammemes_list), np.vstack(chars_list), np.hstack(y_list)

    def train(self, filenames: List[str], validation_size: int=5,
              validation_verbosity: int=5, dump_model_freq: int=10,
              save_path: str=GENERATOR_LSTM_MODEL_PATH, start_epoch: int=0,
              big_epochs: int=10) -> None:
        """
        Обучение модели.
        
        :param filenames: имена файлов с морфоразметкой.
        :param validation_size: размер val выборки.
        :param validation_verbosity: каждый validation_verbosity-шаг делается валидация.
        :param dump_model_freq: каждый dump_model_freq-шаг сохраняется модель.
        :param save_path: путь, куда сохранять модель.
        :param start_epoch: эпоха, с которой надо начать.
        :param big_epochs: количество полных проходов по корпусу
        """
        batch_generator = BatchGenerator(filenames,
                                         batch_size=self.external_batch_size,
                                         embedding_size=self.embedding_size,
                                         softmax_size=self.softmax_size,
                                         sentence_maxlen=self.sentence_maxlen,
                                         word_form_vocabulary=self.word_form_vocabulary,
                                         grammeme_vectorizer=self.grammeme_vectorizer,
                                         max_word_len=self.max_word_len)

        lemmas_val, grammemes_val, chars_val, y_val = \
            LSTMGenerator.__get_validation_data(batch_generator, validation_size)
        for big_epoch in range(big_epochs):
            print('------------Big Epoch {}------------'.format(big_epoch))
            for epoch, (lemmas, grammemes, chars, y) in enumerate(batch_generator):
                if epoch < start_epoch:
                    continue
                if epoch < validation_size:
                    continue
                self.model.fit([lemmas, grammemes, chars], y, batch_size=self.nn_batch_size, epochs=1, verbose=2)

                if epoch != 0 and epoch % validation_verbosity == 0:
                    print('val loss:', self.model.evaluate([lemmas_val, grammemes_val, chars_val],
                                                           y_val, batch_size=self.nn_batch_size * 2, verbose=0))

                indices = [self.word_form_vocabulary.get_sequence_end_index()]
                for _ in range(10):
                    indices.append(self._sample(self.predict(indices)))
                sentence = [self.word_form_vocabulary.get_word_form_by_index(index) for index in indices]
                print('Sentence', str(big_epoch), str(epoch), end=': ')
                for word in sentence[::-1]:
                    print(word.text, end=' ')
                print()

                if epoch != 0 and epoch % dump_model_freq == 0:
                    self.save(save_path)

    def predict(self, word_indices: List[int]) -> np.array:
        """
        Предсказание вероятностей следующего слова.
        
        :param word_indices: индексы предыдущих слов.
        :return: проекция языковой модели (вероятности следующего слова).
        """
        if len(word_indices) == 0:
            return np.full(self.softmax_size, 1.0 / self.softmax_size, dtype=np.float)

        cur_sent = [self.word_form_vocabulary.get_word_form_by_index(ind) for ind in word_indices]

        x_lemmas = np.zeros((1, len(cur_sent)))
        x_grammemes = np.zeros((1, len(cur_sent), self.grammeme_vectorizer.grammemes_count()))
        x_chars = np.zeros((1, len(cur_sent), self.max_word_len))

        lemmas_vector, grammemes_vector, chars_vector =\
            BatchGenerator.get_sample(cur_sent, self.embedding_size, self.max_word_len,
                                      word_form_vocabulary=self.word_form_vocabulary,
                                      grammeme_vectorizer=self.grammeme_vectorizer)

        x_lemmas[0, -len(cur_sent):] = lemmas_vector
        x_grammemes[0, -len(cur_sent):] = grammemes_vector
        x_chars[0, -len(cur_sent):] = chars_vector
        prob = self.model.predict([x_lemmas, x_grammemes, x_chars], verbose=0)[0]
        return prob

    @staticmethod
    def _sample(prob: np.array, temperature: float=1.0) -> int:
        """
        Выбор слова по набору вероятностей с заданной температурой (распределение Больцмана).
        
        :param prob: вероятности.
        :param temperature: температура.
        :return: индекс итогового слова.
        """
        prob = prob[:-1]  # Для исключения неизвестных слов.
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.random.choice(len(prob), p=prob)


class LSTMModelContainer(ModelContainer):
    """
    Контейнер для языковой модели на основе LSTM.
    """
    def __init__(self, model_path=GENERATOR_LSTM_MODEL_PATH,
                 word_form_vocab_dump_path: str=GENERATOR_WORD_FORM_VOCAB_PATH,
                 gram_dump_path: str=GENERATOR_GRAM_VECTORS, cut_context=10):
        self.lstm = LSTMGenerator(lstm_units=512, dense_units=256, embedding_size=10000, softmax_size=50000)
        self.lstm.prepare(list(), word_form_vocab_dump_path, gram_dump_path)
        self.lstm.load(model_path)
        self.cut_context = cut_context

    def get_model(self, word_indices: List[int]) -> np.array:
        return self.lstm.predict(word_indices[-self.cut_context:])
