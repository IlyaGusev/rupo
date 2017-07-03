# -*- coding: utf-8 -*-
# Авторы: Анастасьев Даниил, Гусев Илья
# Описание: Модуль рекуррентой сети для генерации языковой модели.

import numpy as np
from typing import List, Tuple
import keras.activations
from keras.models import model_from_json
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dense, Merge, LSTM, SpatialDropout1D, Masking
from keras import backend as K

from rupo.generate.word_form import WordForm
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.generate.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.model_container import ModelContainer
from rupo.settings import GENERATOR_LSTM_MODEL_PATH


def hard_tanh(x):
    return K.minimum(K.maximum(x, K.constant(-1)), K.constant(1))
keras.activations.hard_tanh = hard_tanh


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """
    def __init__(self, filenames: List[str], batch_size: int, softmax_size: int, sentence_maxlen: int,
                 word_form_vocabulary: WordFormVocabulary, grammeme_vectorizer: GrammemeVectorizer):
        """
        :param filenames: имена файлов с морфоразметкой.
        :param batch_size: размер набора семплов.
        :param softmax_size: размер выхода softmax-слоя (=размер итогового набора вероятностей)
        :param sentence_maxlen: маскимальная длина куска предложения.
        :param word_form_vocabulary: словарь словофрм.
        :param grammeme_vectorizer: векторизатор граммем.
        """
        self.filenames = filenames  # type: List[str]
        self.batch_size = batch_size  # type: int
        self.softmax_size = softmax_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_form_vocabulary = word_form_vocabulary  # type: WordFormVocabulary
        self.grammeme_vectorizer = grammeme_vectorizer  # type: GrammemeVectorizer
        assert self.word_form_vocabulary.sorted

    def __generate_seqs(self, sentences: List[List[WordForm]]) -> Tuple[List[List[WordForm]], List[WordForm]]:
        """
        Генерация семплов.
        
        :param sentences: куски предложений из словоформ.
        :return: пары (<семпл из словоформ>, следующая за ним словоформа (ответ)).
        """
        seqs, next_words = [], []
        for sentence in sentences:
            # Разворот для генерации справа налево.
            sentence = sentence[::-1]
            for i in range(1, len(sentence)):
                word_form = sentence[i]
                # Если следующяя слоформа не из предсказываемых, пропускаем семпл.
                if self.word_form_vocabulary.get_word_form_index(word_form) >= self.softmax_size:
                    continue
                seqs.append(sentence[max(0, i-self.sentence_maxlen):i])
                next_words.append(word_form)
        return seqs, next_words

    def __to_tensor(self, sentences: List[List[WordForm]], next_words: List[WordForm]) -> \
            Tuple[np.array, np.array, np.array]:
        """
        Перевод семплов из словоформ в индексы словоформ, поиск грамматических векторов по индексу.
        
        :param sentences: семплы из словоформ.
        :param next_words: овтеты-словоформы на семплы.
        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        n_samples = len(sentences)
        max_len = max(len(sent) for sent in sentences)
        x_words = np.zeros((n_samples, max_len), dtype=np.int)
        x_grammemes = np.zeros((n_samples, max_len, self.grammeme_vectorizer.grammemes_count()), dtype=np.int)
        y = np.zeros(n_samples, dtype=np.int)
        for i in range(n_samples):
            sentence = sentences[i]
            next_word = next_words[i]
            # Функция get_word_form_index_min нужна здесь для игнорирования неиспользуемых словоформ.
            x_words[i, -len(sentence):] = [self.word_form_vocabulary.get_word_form_index_min(x, self.softmax_size)
                                           for x in sentence]
            x_grammemes[i, -len(sentence):] = [self.grammeme_vectorizer.vectors[x.gram_vector_index] for x in sentence]
            y[i] = self.word_form_vocabulary.get_word_form_index(next_word)
        return x_words, x_grammemes, y

    def __iter__(self):
        """
        Получние очередного батча.
        
        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        sentences = [[]]
        for filename in self.filenames:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sentences.append([])
                    else:
                        word, lemma, pos, tags = line.split('\t')[:4]
                        word = word.lower()
                        lemma = lemma.lower()
                        gram_vector_index = self.grammeme_vectorizer.name_to_index[pos+"#"+tags]
                        sentences[-1].append(WordForm(lemma + '_' + pos, gram_vector_index, word))
                    if len(sentences) >= self.batch_size:
                        sentences, next_words = self.__generate_seqs(sentences)
                        yield self.__to_tensor(sentences, next_words)
                        sentences = [[]]
                sentences, next_words = self.__generate_seqs(sentences)
                yield self.__to_tensor(sentences, next_words)
                sentences = [[]]


class LSTMGenerator:
    """
    Языковая модель на основе двухуровневой LSTM RNN.
    """
    def __init__(self, softmax_size: int=60000, external_batch_size: int=10000,
                 nn_batch_size: int=768, sentence_maxlen: int=10, lstm_units=368,
                 embeddings_dimension=150, grammeme_dense_units=25):
        """
        :param softmax_size: размер выхода softmax-слоя (=размер итогового набора вероятностей)
        :param external_batch_size: размер набора семплов для BatchGenerator'а.
        :param nn_batch_size: размер набора семплов для обучения.
        :param sentence_maxlen: маскимальная длина куска предложения.
        """
        self.softmax_size = softmax_size  # type: int
        self.external_batch_size = external_batch_size  # type: int
        self.nn_batch_size = nn_batch_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_form_vocabulary = None  # type: WordFormVocabulary
        self.grammeme_vectorizer = None  # type: GrammemeVectorizer
        self.lstm_units = lstm_units  # type: int
        self.embeddings_dimension = embeddings_dimension  # type: int
        self.grammeme_dense_units = grammeme_dense_units  # type: int
        self.model = None  # type: Model

    def prepare(self, filenames: List[str]=list()) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря словоформ по корпусу.
        
        :param filenames: имена файлов с морфоразметкой.
        """
        self.grammeme_vectorizer = GrammemeVectorizer()
        if self.grammeme_vectorizer.is_empty():
            for filename in filenames:
                self.grammeme_vectorizer.collect_grammemes(filename)
            for filename in filenames:
                self.grammeme_vectorizer.collect_possible_vectors(filename)
            self.grammeme_vectorizer.save()

        self.word_form_vocabulary = WordFormVocabulary()
        if self.word_form_vocabulary.is_empty():
            for filename in filenames:
                self.word_form_vocabulary.load_from_corpus(filename, self.grammeme_vectorizer)
            self.word_form_vocabulary.sort()
            self.word_form_vocabulary.save()
        # self.word_form_vocabulary.inflate_vocab(self.softmax_size)

    def load(self, model_filename: str) -> None:
        """
        Загрузка модели.
        
        :param model_filename: файл с моделью.
        """
        self.model = load_model(model_filename)

    def load_with_weights(self, json_filename: str, weights_filename: str) -> None:
        """
        Загрузка модели из json описания и файла с весами.
        
        :param json_filename: json описание.
        :param weights_filename: файл с весам.
        """
        json_string = open(json_filename, 'r', encoding='utf8').readline()
        self.model = model_from_json(json_string)
        self.model.load_weights(weights_filename)

    def build(self):
        """
        Описание модели.
        """
        words = Input(shape=(None,), name='words')

        words_embedding = SpatialDropout1D(.3)(Embedding(self.softmax_size + 1, self.embeddings_dimension, name='embeddings')(words))
        grammemes_input = Input(shape=(None, self.grammeme_vectorizer.grammemes_count()), name='grammemes')
        grammemes_layer = Masking(mask_value=0.)(grammemes_input)
        grammemes_layer = Dense(self.grammeme_dense_units, activation=hard_tanh)(grammemes_layer)
        grammemes_layer = Dense(self.grammeme_dense_units, activation=hard_tanh)(grammemes_layer)
        layer = Merge(mode='concat', name='LSTM_input')([words_embedding, grammemes_layer])
        layer = LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2, return_sequences=True, name='LSTM_1')(layer)
        layer = LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2, return_sequences=False, name='LSTM_2')(layer)

        output = Dense(self.softmax_size + 1, activation='softmax')(layer)

        self.model = Model(inputs=[words, grammemes_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        print(self.model.summary())

    def train(self, filenames: List[str]) -> None:
        """
        Обучение модели.
        
        :param filenames: имена файлов с морфоразметкой.
        """
        batch_generator = BatchGenerator(filenames,
                                         batch_size=self.external_batch_size,
                                         softmax_size=self.softmax_size,
                                         sentence_maxlen=self.sentence_maxlen,
                                         word_form_vocabulary=self.word_form_vocabulary,
                                         grammeme_vectorizer=self.grammeme_vectorizer)
        for big_epoch in range(0, 1000):
            print('------------Big Epoch {}------------'.format(big_epoch))
            for epoch, (X1, X2, y) in enumerate(batch_generator):
                self.model.fit([X1, X2], y, batch_size=self.nn_batch_size, epochs=1, verbose=2)
                indices = [np.random.randint(0, self.softmax_size)]
                for i in range(10):
                    indices.append(self._sample(self.predict(indices)))
                sentence = [self.word_form_vocabulary.get_word_form_by_index(index) for index in indices]
                print('Sentence', str(big_epoch), str(epoch), end=': ')
                for word in sentence[::-1]:
                    print(word.text, end=' ')
                print()

                if epoch != 0 and epoch % 10 == 0:
                    self.model.save(GENERATOR_LSTM_MODEL_PATH)

    def predict(self, word_indices: List[int]) -> np.array:
        """
        Предсказание вероятностей следующего слова.
        
        :param word_indices: индексы предыдущих слов.
        :return: проекция языковой модели (вероятности следующего слова).
        """
        if len(word_indices) == 0:
            return np.full(self.softmax_size, 1.0/self.softmax_size, dtype=np.float)
        cur_sent = [self.word_form_vocabulary.get_word_form_by_index(ind) for ind in word_indices]
        x_emb = np.zeros((1, len(cur_sent)))
        x_gram = np.zeros((1, len(cur_sent), self.grammeme_vectorizer.grammemes_count()))
        for index, word in enumerate(cur_sent):
            x_emb[0, index] = self.word_form_vocabulary.get_word_form_index_min(word, self.softmax_size)
            x_gram[0, index] = self.grammeme_vectorizer.vectors[word.gram_vector_index]
        prob = self.model.predict([x_emb, x_gram], verbose=0)[0]
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
    def __init__(self, model_path=GENERATOR_LSTM_MODEL_PATH):
        self.lstm = LSTMGenerator(softmax_size=50000)
        self.lstm.prepare()
        self.lstm.load(model_path)

    def get_model(self, word_indices: List[int]) -> np.array:
        return self.lstm.predict(word_indices)


