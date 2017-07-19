# -*- coding: utf-8 -*-
# Авторы: Анастасьев Даниил, Гусев Илья
# Описание: Модуль рекуррентой сети для генерации языковой модели.

import numpy as np
from typing import List, Tuple
from itertools import islice
import keras.activations
from keras.models import model_from_json
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dense, Merge, LSTM, SpatialDropout1D, Masking, BatchNormalization, Activation
from keras import backend as K

from rupo.generate.word_form import WordForm
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.generate.corpora_information_loader import CorporaInformationLoader, SEQ_END_WF
from rupo.generate.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.model_container import ModelContainer
from rupo.generate.tqdm_open import tqdm_open
from rupo.settings import GENERATOR_LSTM_MODEL_PATH, GENERATOR_WORD_FORM_VOCAB_PATH, GENERATOR_GRAM_VECTORS


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """
    def __init__(self, filenames: List[str], batch_size: int, 
                 embedding_size: int, softmax_size: int, sentence_maxlen: int,
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
        self.embedding_size = embedding_size # type: int
        self.softmax_size = softmax_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_form_vocabulary = word_form_vocabulary  # type: WordFormVocabulary
        self.grammeme_vectorizer = grammeme_vectorizer  # type: GrammemeVectorizer

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
                # Если следующая словооформа не из предсказываемых, пропускаем её.
                if self.word_form_vocabulary.lemma_indices[word_form] >= self.softmax_size:
                    continue
                seqs.append(sentence[max(0, i-self.sentence_maxlen) : i])
                next_words.append(word_form)
        return seqs, next_words

    def __to_tensor(self, sentences: List[List[WordForm]], next_words: List[WordForm]) -> \
            Tuple[np.array, np.array, np.array]:
        """
        Перевод семплов из словоформ в индексы словоформ, поиск грамматических векторов по индексу.
        
        :param sentences: семплы из словоформ.
        :param next_words: следующие за последовательностями из sentences слова.
        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        n_samples = len(sentences)
        max_len = max(len(sent) for sent in sentences)
        lemmas = np.zeros((n_samples, max_len), dtype=np.int)
        grammemes = np.zeros((n_samples, max_len, self.grammeme_vectorizer.grammemes_count()), dtype=np.int)
        y = np.zeros(n_samples, dtype=np.int)
        for i in range(n_samples):
            sentence = sentences[i]
            next_word = next_words[i]
            lemmas[i, -len(sentence):] = [min(self.word_form_vocabulary.lemma_indices[x], self.embedding_size) 
                                          for x in sentence]
            grammemes[i, -len(sentence):] = [self.grammeme_vectorizer.get_vector_by_index(x.gram_vector_index)
                                             for x in sentence]
            y[i] = min(self.word_form_vocabulary.word_form_indices[next_word], self.softmax_size)
        return lemmas, grammemes, y

    def __iter__(self):
        """
        Получение очередного батча.
        
        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        sentences = [[]]
        for filename in self.filenames:
            with tqdm_open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sentences[-1].append(SEQ_END_WF)
                        sentences.append([])
                    else:
                        word, lemma, pos, tags = line.split('\t')[:4]
                        word, lemma = word.lower(), lemma.lower()
                        gram_vector_index = self.grammeme_vectorizer.name_to_index[pos+"#"+tags]
                        sentences[-1].append(WordForm(lemma + '_' + pos, gram_vector_index, word))
                    if len(sentences) >= self.batch_size:
                        sentences, next_words = self.__generate_seqs(sentences)
                        yield self.__to_tensor(sentences, next_words)
                        sentences = [[]]


class LSTMGenerator:
    """
    Языковая модель на основе двухуровневой LSTM RNN.
    """
    def __init__(self, embedding_size: int=30000, softmax_size: int=60000, 
                 external_batch_size: int=10000, nn_batch_size: int=768, 
                 sentence_maxlen: int=10, lstm_units=368, embeddings_dimension: int=150, 
                 grammeme_dense_units: List[int]=[35, 15], dense_units: int=256):
        """
        :param embeddings_size: размер входного слоя (=размер словаря)
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
        self.model = None  # type: Model

    def prepare(self, filenames: List[str]=list(),
                word_form_vocab_dump_path: str=GENERATOR_WORD_FORM_VOCAB_PATH,
                gram_dump_path: str=GENERATOR_GRAM_VECTORS) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря словоформ по корпусу.
        
        :param filenames: имена файлов с морфоразметкой.
        :param word_form_vocab_dump_path: путь к дампу словаря словоформ.
        :param gram_dump_path: путь к векторам грамматических значений.
        """
        self.grammeme_vectorizer = GrammemeVectorizer(gram_dump_path)
        self.word_form_vocabulary = WordFormVocabulary(word_form_vocab_dump_path)
        if self.grammeme_vectorizer.is_empty() or self.word_form_vocabulary.is_empty():
            loader = CorporaInformationLoader()

            self.word_form_vocabulary, self.grammeme_vectorizer = loader.parse_corpora(filenames)
            self.grammeme_vectorizer.save()
            self.word_form_vocabulary.save()

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
        # Вход лемм
        lemmas = Input(shape=(None,), name='lemmas')
        lemmas_embedding = Embedding(self.embedding_size + 1, self.embeddings_dimension, name='embeddings')(lemmas)
        lemmas_embedding = SpatialDropout1D(.3)(lemmas_embedding)

        # Вход граммем
        grammemes_input = Input(shape=(None, self.grammeme_vectorizer.grammemes_count()), name='grammemes')
        grammemes_layer = Masking(mask_value=0.)(grammemes_input)
        for grammeme_dense_layer_units in self.grammeme_dense_units:
            grammemes_layer = Dense(grammeme_dense_layer_units, activation='relu')(grammemes_layer)

        layer = Merge(mode='concat', name='LSTM_input')([lemmas_embedding, grammemes_layer])
        layer = LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2, return_sequences=True, name='LSTM_1')(layer)
        layer = LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2, return_sequences=False, name='LSTM_2')(layer)

        layer = Dense(self.dense_units)(layer)
        layer = BatchNormalization()(layer)
        layer = Activation('relu')(layer)

        output = Dense(self.softmax_size + 1, activation='softmax')(layer)

        self.model = Model(inputs=[lemmas, grammemes_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        print(self.model.summary())

    @staticmethod
    def __get_validation_data(batch_generator, size):
        """
        Берет первые size батчей и batch_generator для валидационной выборки
        """
        lemmas_list, grammemes_list, y_list = [], [], []
        for lemmas, grammemes, y in islice(batch_generator, size):
            lemmas_list.append(lemmas)
            grammemes_list.append(grammemes)
            y_list.append(y)
        return np.vstack(lemmas_list), np.vstack(grammemes_list), np.hstack(y_list)

    def train(self, filenames: List[str], validation_size: int=5, 
            validation_verbosity: int=5, dump_model_freq: int = 10) -> None:
        """
        Обучение модели.
        
        :param filenames: имена файлов с морфоразметкой.
        """
        batch_generator = BatchGenerator(filenames,
                                         batch_size=self.external_batch_size,
                                         embedding_size=self.embedding_size,
                                         softmax_size=self.softmax_size,
                                         sentence_maxlen=self.sentence_maxlen,
                                         word_form_vocabulary=self.word_form_vocabulary,
                                         grammeme_vectorizer=self.grammeme_vectorizer)

        lemmas_val, grammemes_val, y_val = LSTMGenerator.__get_validation_data(batch_generator, validation_size)
        for big_epoch in range(0, 1000):
            print('------------Big Epoch {}------------'.format(big_epoch))
            for epoch, (lemmas, grammemes, y) in enumerate(batch_generator):
                if epoch < validation_size:
                    continue
                self.model.fit([lemmas, grammemes], y, batch_size=self.nn_batch_size, epochs=1, verbose=2)

                if epoch != 0 and epoch % validation_verbosity == 0:
                    print('val loss:', self.model.evaluate([lemmas_val, grammemes_val], 
                        y_val, batch_size=self.nn_batch_size * 2, verbose=0))

                indices = [self.word_form_vocabulary.get_sequence_end_index(SEQ_END_WF)]
                for _ in range(10):
                    indices.append(self._sample(self.predict(indices)))
                sentence = [self.word_form_vocabulary.get_word_form_by_index(index) for index in indices]
                print('Sentence', str(big_epoch), str(epoch), end=': ')
                for word in sentence[::-1]:
                    print(word.text, end=' ')
                print()

                if epoch != 0 and epoch % dump_model_freq == 0:
                    self.model.save(GENERATOR_LSTM_MODEL_PATH)

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
        for index, word in enumerate(cur_sent):
            x_lemmas[0, index] = self.word_form_vocabulary.get_word_form_index_min(word, self.softmax_size)
            x_grammemes[0, index] = self.grammeme_vectorizer.vectors[word.gram_vector_index]
        prob = self.model.predict([x_lemmas, x_grammemes], verbose=0)[0]
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
                 gram_dump_path: str=GENERATOR_GRAM_VECTORS):
        self.lstm = LSTMGenerator(softmax_size=50000)
        self.lstm.prepare(list(), word_form_vocab_dump_path, gram_dump_path)
        self.lstm.load(model_path)

    def get_model(self, word_indices: List[int]) -> np.array:
        return self.lstm.predict(word_indices)
