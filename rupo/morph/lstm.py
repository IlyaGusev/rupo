import numpy as np
from typing import List, Tuple
from collections import Counter
from itertools import islice
from keras.models import model_from_json,  Model, load_model
from keras.layers import Input, Embedding, Dense, LSTM, SpatialDropout1D, Masking, \
    BatchNormalization, Activation, concatenate, Bidirectional, TimeDistributed, Dropout
from keras.optimizers import Adam
from russian_tagsets import converters
import tensorflow as tf

from rupo.generate.word_form import WordForm
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.generate.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.tqdm_open import tqdm_open
from rupo.settings import GENERATOR_LSTM_MODEL_PATH, GENERATOR_WORD_FORM_VOCAB_PATH, GENERATOR_GRAM_VECTORS
import pymorphy2
from rupo.morph.loader import WordVocabulary, Loader, process_tag


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """

    def __init__(self, filenames: List[str], batch_size: int,
                 input_size: int, sentence_maxlen: int, word_vocabulary: WordVocabulary,
                 grammeme_vectorizer: GrammemeVectorizer):
        self.filenames = filenames  # type: List[str]
        self.batch_size = batch_size  # type: int
        self.input_size = input_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_vocabulary = word_vocabulary  # type: WordVocabulary
        self.grammeme_vectorizer = grammeme_vectorizer  # type: GrammemeVectorizer
        self.morph = pymorphy2.MorphAnalyzer()

    def __to_tensor(self, sentences: List[List[WordForm]]) -> Tuple[np.array, np.array, np.array]:
        n_samples = sum([len(sentence) for sentence in sentences])
        words = np.zeros((n_samples, self.sentence_maxlen), dtype=np.int)
        grammemes = np.zeros((n_samples, self.sentence_maxlen, self.grammeme_vectorizer.size()), dtype=np.float)
        y = np.zeros((n_samples, self.sentence_maxlen), dtype=np.int)
        i = 0
        for sentence in sentences:
            if len(sentence) <= 1:
                continue
            sentence = sentence[:self.sentence_maxlen]
            texts = [x.text for x in sentence]
            word_indices, gram_vectors = self.get_sample(texts, self.morph, self.grammeme_vectorizer,
                                                         self.word_vocabulary, self.input_size)
            assert len(word_indices) == len(sentence)
            assert len(gram_vectors) == len(sentence)
            words[i, -len(sentence):] = word_indices
            grammemes[i, -len(sentence):] = gram_vectors
            y[i, -len(sentence):] = [word.gram_vector_index + 1 for word in sentence]
            i += 1
        y = y.reshape(y.shape[0], y.shape[1], 1)
        return words, grammemes,  y

    @staticmethod
    def get_sample(sentence: List[str], morph, grammeme_vectorizer, word_vocabulary, input_size):
        to_ud = converters.converter('opencorpora-int', 'ud14')
        gram_vectors = []
        for word in sentence:
            gram_value_indices = np.zeros(grammeme_vectorizer.size())
            for parse in morph.parse(word):
                pos, gram = process_tag(to_ud, parse.tag, word)
                gram_value_indices[grammeme_vectorizer.get_index_by_name(pos + "#" + gram)] = parse.score
            gram_vectors.append(gram_value_indices)
        word_indices = [min(word_vocabulary.word_to_index[word.lower()]
                            if word in word_vocabulary.word_to_index else input_size,
                            input_size) for word in sentence]
        return word_indices, gram_vectors

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
                        sentences.append([])
                    else:
                        word, lemma, pos, tags = line.split('\t')[:4]
                        word, lemma = word.lower(), lemma.lower() + '_' + pos
                        tags = "|".join(sorted(tags.split("|")))
                        gram_vector_index = self.grammeme_vectorizer.name_to_index[pos + "#" + tags]
                        sentences[-1].append(WordForm(lemma, gram_vector_index, word))
                    if len(sentences) >= self.batch_size:
                        yield self.__to_tensor(sentences)
                        sentences = [[]]


class LSTMMorphoAnalysis:
    def __init__(self, input_size: int=5000, external_batch_size: int=5000, nn_batch_size: int=256,
                 sentence_maxlen: int=30, lstm_units=256, embeddings_dimension: int=200, dense_units: int=128):
        self.input_size = input_size  # type: int
        self.external_batch_size = external_batch_size  # type: int
        self.nn_batch_size = nn_batch_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.word_form_vocabulary = None  # type: WordFormVocabulary
        self.grammeme_vectorizer = None  # type: GrammemeVectorizer
        self.lstm_units = lstm_units  # type: int
        self.embeddings_dimension = embeddings_dimension  # type: int
        self.dense_units = dense_units  # type: int
        self.model = None  # type: Model
        self.morph = pymorphy2.MorphAnalyzer()

    def prepare(self, word_vocab_dump_path: str, gram_dump_path: str, filenames: List[str]=None) -> None:
        """
        Подготовка векторизатора грамматических значений и словаря слов по корпусу.

        :param word_vocab_dump_path: путь к дампу словаря слов.
        :param gram_dump_path: путь к векторам грамматических значений.
        :param filenames: имена файлов с морфоразметкой.
        """
        self.grammeme_vectorizer = GrammemeVectorizer(gram_dump_path)
        self.word_vocabulary = WordVocabulary(word_vocab_dump_path)
        if self.grammeme_vectorizer.is_empty() or self.word_vocabulary.is_empty():
            loader = Loader(gram_dump_path, word_vocab_dump_path)
            self.grammeme_vectorizer, self.word_vocabulary = loader.parse_corpora(filenames)
            self.grammeme_vectorizer.save()
            self.word_vocabulary.save()

    def load(self, model_filename: str) -> None:
        """
        Загрузка модели.

        :param model_filename: файл с моделью.
        """
        self.model = load_model(model_filename)

    def build(self):
        """
        Описание модели.
        """
        # Вход лемм
        words = Input(shape=(None,), name='words')
        words_embedding = Embedding(self.input_size + 1, self.embeddings_dimension, name='embeddings')(words)

        # Вход граммем
        grammemes_input = Input(shape=(None, self.grammeme_vectorizer.size()), name='grammemes')

        layer = concatenate([words_embedding, grammemes_input], name="LSTM_input")
        layer = Bidirectional(LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2,
                                   return_sequences=True, name='LSTM_1'))(layer)
        layer = Bidirectional(LSTM(self.lstm_units, dropout=.2, recurrent_dropout=.2,
                                   return_sequences=True, name='LSTM_2'))(layer)

        layer = TimeDistributed(Dense(self.dense_units))(layer)
        layer = TimeDistributed(Dropout(.2))(layer)
        layer = TimeDistributed(BatchNormalization())(layer)
        layer = TimeDistributed(Activation('relu'))(layer)

        output = TimeDistributed(Dense(self.grammeme_vectorizer.size() + 1, activation='softmax'))(layer)

        self.model = Model(inputs=[words, grammemes_input], outputs=[output])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        print(self.model.summary())

    @staticmethod
    def __get_validation_data(batch_generator, size):
        """
        Берет первые size батчей из batch_generator для валидационной выборки
        """
        words_list, grammemes_list, y_list = [], [], []
        for words, grammemes, y in islice(batch_generator, size):
            words_list.append(words)
            grammemes_list.append(grammemes)
            y_list.append(y)
        return np.vstack(words_list), np.vstack(grammemes_list), np.vstack(y_list)

    def train(self, filenames: List[str], save_path: str, validation_size: int = 2,
              validation_verbosity: int = 5, dump_model_freq: int = 5) -> None:
        """
        Обучение модели.

        :param filenames: имена файлов с морфоразметкой.
        :param validation_size: размер val выборки.
        :param validation_verbosity: каждый validation_verbosity-шаг делается валидация.
        :param dump_model_freq: каждый dump_model_freq-шаг сохраняется модель.
        """
        batch_generator = BatchGenerator(filenames,
                                         batch_size=self.external_batch_size,
                                         grammeme_vectorizer=self.grammeme_vectorizer,
                                         word_vocabulary=self.word_vocabulary,
                                         input_size=self.input_size,
                                         sentence_maxlen=self.sentence_maxlen)

        words_val, grammemes_val, y_val = LSTMMorphoAnalysis.__get_validation_data(batch_generator, validation_size)
        for big_epoch in range(0, 1000):
            print('------------Big Epoch {}------------'.format(big_epoch))
            for epoch, (words, grammemes, y) in enumerate(batch_generator):
                if epoch < validation_size:
                    continue
                self.model.fit([words, grammemes], y, batch_size=self.nn_batch_size, epochs=1, verbose=2)

                if epoch != 0 and epoch % validation_verbosity == 0:
                    print('val loss:',
                          self.model.evaluate([words_val, grammemes_val],
                                              y_val, batch_size=self.nn_batch_size * 2, verbose=0))

                if epoch != 0 and epoch % dump_model_freq == 0:
                    self.model.save(save_path)

    def predict(self, sentence: List[str]):
        word_indices, gram_vectors = BatchGenerator.get_sample(sentence, self.morph, self.grammeme_vectorizer,
                                                               self.word_vocabulary, self.input_size)
        words = np.zeros((1, self.sentence_maxlen), dtype=np.int)
        grammemes = np.zeros((1, self.sentence_maxlen, self.grammeme_vectorizer.size()), dtype=np.float)
        words[0, -len(sentence):] = word_indices
        grammemes[0, -len(sentence):] = gram_vectors
        answer = []
        for grammeme_probs in self.model.predict([words, grammemes])[0][-len(sentence):]:
            num = np.argmax(grammeme_probs[1:])
            answer.append(self.grammeme_vectorizer.get_name_by_index(num))
        return answer
