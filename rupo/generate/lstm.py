import pickle
import numpy as np

from time import strftime, localtime
from typing import List
from contextlib import redirect_stdout
from keras.models import model_from_json
from keras.callbacks import Callback
from keras.models import Model
from keras.layers import Input, Activation, Embedding, Dense, Merge, LSTM, SpatialDropout1D, Masking
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from rupo.generate.word_form import WordForm
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.generate.grammeme_vectorizer import GrammemeVectorizer

from rupo.settings import GENERATOR_MODEL_DESCRIPTION, GENERATOR_MODEL_WEIGHTS, GENERATOR_LEMMA_VOCAB_PATH, \
    GENERATOR_TAGS_VECTORS


class BatchGenerator:
    def __init__(self, filename, batch_size, softmax_size, sentence_maxlen,
                 word_form_vocabulary: WordFormVocabulary, grammeme_vectorizer: GrammemeVectorizer):
        self.filename = filename
        self.batch_size = batch_size
        self.softmax_size = softmax_size
        self.sentence_maxlen = sentence_maxlen
        self.word_form_vocabulary = word_form_vocabulary
        self.grammeme_vectorizer = grammeme_vectorizer
        assert self.word_form_vocabulary.sorted

    def __generate_seqs(self, sentences):
        seqs, next_words = [], []
        for sentence in sentences:
            sentence = sentence[::-1]
            for i in range(1, len(sentence)):
                word_form = sentence[i]
                if self.word_form_vocabulary.get_word_form_index(word_form) >= self.softmax_size:
                    continue
                seqs.append(sentence[max(0, i-self.sentence_maxlen):i])
                next_words.append(word_form)
        return seqs, next_words

    def __to_tensor(self, sentences):
        sentences, next_words = self.__generate_seqs(sentences)
        n_samples = len(sentences)
        max_len = max(len(sent) for sent in sentences)
        x_emb = np.zeros((n_samples, max_len), dtype=np.int)
        x_grammemes = np.zeros((n_samples, max_len, self.grammeme_vectorizer.grammemes_count()), dtype=np.int)
        y = np.zeros(n_samples, dtype=np.int)
        for i in range(n_samples):
            sentence = sentences[i]
            next_word = next_words[i]
            x_emb[i, -len(sentence):] = [self.word_form_vocabulary.get_word_form_index_min(x, self.softmax_size) for x in sentence]
            x_grammemes[i, -len(sentence):] = [self.grammeme_vectorizer.vectors[x.gram_vector_index] for x in sentence]
            y[i] = self.word_form_vocabulary.get_word_form_index_min(next_word, self.softmax_size)
        return x_emb, x_grammemes, y

    def to_tensor(self, words_indices):
        return self.__to_tensor([self.word_form_vocabulary.word_forms[i] for i in words_indices])

    def __iter__(self):
        sentences = [[]]
        with open(self.filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    sentences.append([])
                else:
                    word, lemma, pos, tags = line.split('\t')
                    gram_vector_index = self.grammeme_vectorizer.name_to_index[pos+"#"+tags]
                    sentences[-1].append(WordForm(lemma + '_' + pos, gram_vector_index, word))
                if len(sentences) >= self.batch_size:
                    yield self.__to_tensor(sentences)
                    sentences = [[]]


class EvalCallback(Callback):
    def __init__(self, name, softmax_size,
                 word_form_vocabulary: WordFormVocabulary,
                 grammeme_vectorizer: GrammemeVectorizer,
                 model):
        self._name = name
        self.softmax_size = softmax_size
        self.word_form_vocabulary = word_form_vocabulary
        self.grammeme_vectorizer = grammeme_vectorizer
        self.model = model
        super(Callback, self).__init__()

    def write(self, message):
        print('[{}] {}'.format(strftime("%H:%M:%S", localtime()), message))

    def _sample(self, prob, temperature=1.0):
        prob = prob[:-1]  # Не хотим предсказывать UNKNOWN_WORD
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.random.choice(len(prob), p=prob)

    def _generate(self):
        cur_sent = [self.word_form_vocabulary.get_word_form_by_index(np.random.randint(0, self.softmax_size))]
        for i in range(10):
            x_emb = np.zeros((1, len(cur_sent)))
            x_gram = np.zeros((1, len(cur_sent), self.grammeme_vectorizer.grammemes_count()))
            for index, word in enumerate(cur_sent):
                x_emb[0, index] = self.word_form_vocabulary.get_word_form_index_min(word, self.softmax_size)
                x_gram[0, index] = self.grammeme_vectorizer.vectors[word.gr_tag]

            preds = self.model.predict([x_emb, x_gram], verbose=0)[0]
            cur_sent.append(self.word_form_vocabulary.get_word_form_by_index(self._sample(preds)))

        print('Sentence', end=': ')
        for word in cur_sent[::-1]:
            print(word.word_form, end=' ')
        print()

    def on_epoch_end(self, epoch, logs={}):
        self._generate()


class LSTMGenerator:
    def __init__(self, softmax_size=60000):
        self.softmax_size = softmax_size
        self.word_form_vocabulary = None
        self.grammeme_vectorizer = None
        self.model = None

    def prepare_data(self, filename):
        self.grammeme_vectorizer = GrammemeVectorizer()
        self.grammeme_vectorizer.collect_grammemes(filename)
        self.grammeme_vectorizer.collect_possible_vectors(filename)
        self.word_form_vocabulary = WordFormVocabulary()
        self.word_form_vocabulary.load_from_corpus(filename, self.grammeme_vectorizer)
        self.word_form_vocabulary.sort()

    def build(self):
        def hard_tanh(x):
            return K.minimum(K.maximum(x, K.constant(-1)), K.constant(1))

        words = Input(shape=(None,), name='words')

        words_embedding = SpatialDropout1D(0.3)(Embedding(self.softmax_size + 1, 150, name='embeddings')(words))
        grammemes_input = Input(shape=(None, self.grammeme_vectorizer.grammemes_count()), name='grammemes')
        grammemes_layer = Masking(mask_value=0.)(grammemes_input)
        grammemes_layer = Dense(25, activation=hard_tanh)(grammemes_layer)
        grammemes_layer = Dense(25, activation=hard_tanh)(grammemes_layer)
        layer = Merge(mode='concat', name='LSTM_input')([words_embedding, grammemes_layer])
        layer = LSTM(368, dropout=.2, recurrent_dropout=.2, return_sequences=True, name='LSTM_1')(layer)
        layer = LSTM(368, dropout=.2, recurrent_dropout=.2, return_sequences=False, name='LSTM_2')(layer)

        output = Dense(self.softmax_size + 1, activation='softmax')(layer)

        self.model = Model(inputs=[words, grammemes_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        print(self.model.summary())

    def train(self, filename):
        batch_generator = BatchGenerator(filename,
                                         batch_size=10000,
                                         softmax_size=self.softmax_size,
                                         sentence_maxlen=10,
                                         word_form_vocabulary=self.word_form_vocabulary,
                                         grammeme_vectorizer=self.grammeme_vectorizer)
        name = 'Lemmatized_hard_tanh'
        with open(name + '_log.txt', 'a') as f:
            with redirect_stdout(f):
                callback = EvalCallback(name, self.softmax_size, self.word_form_vocabulary,
                                        self.grammeme_vectorizer, self.model)
                for big_epoch in range(1000):
                    print('------------Big Epoch {}------------'.format(big_epoch))
                    for epoch, (X1, X2, y) in enumerate(batch_generator):
                        self.model.fit([X1, X2], y, batch_size=768, epochs=1, verbose=2, callbacks=[callback])
                        if epoch != 0 and epoch % 10 == 0:
                            self.model.save_weights(name + '_model.h5')
                        f.flush()

    def predict(self, word_indices: List[int]):
        if len(word_indices) == 0:
            return np.full(self.softmax_size, 1.0/self.softmax_size, dtype=np.float)
        cur_sent = [self.word_form_vocabulary.get_word_by_index(ind) for ind in word_indices]
        x_emb = np.zeros((1, len(cur_sent)))
        x_gram = np.zeros((1, len(cur_sent), self.grammeme_vectorizer.grammemes_count()))
        for index, word in enumerate(cur_sent):
            x_emb[0, index] = self.word_form_vocabulary.get_word_form_index_min(word, self.softmax_size)
            x_gram[0, index] = self.grammeme_vectorizer.vectors[word.gr_tag]
        return self.model.predict([x_emb, x_gram], verbose=0)[0]


class LSTMModelContainer(object):
    def __init__(self):
        json_string = open(GENERATOR_MODEL_DESCRIPTION, 'r', encoding='utf8').readline()
        self.model = model_from_json(json_string)
        self.model.load_weights(GENERATOR_MODEL_WEIGHTS)
        self.lemmatized_vocabulary = pickle.load(open(GENERATOR_LEMMA_VOCAB_PATH, "rb"))
        self.num_of_words = len(self.lemmatized_vocabulary.lemmatizedWords)
        self.index2tags_vector = pickle.load(open(GENERATOR_TAGS_VECTORS, "rb"))
        self.GRAMMEMES_COUNT = 54
        self.SOFTMAX_SIZE = 50000

    def __get_word_index(self, word):
        return min(self.lemmatized_vocabulary.get_word_form_index(word), self.SOFTMAX_SIZE)

    def get_model(self, word_indices: List[int]) -> np.array:
        if len(word_indices) == 0:
            return np.full(self.num_of_words, 1 / self.num_of_words, dtype=np.float)
        cur_sent = [self.lemmatized_vocabulary.get_word_by_index(ind) for ind in word_indices]
        X_emb = np.zeros((1, len(cur_sent)))
        X_gr = np.zeros((1, len(cur_sent), self.GRAMMEMES_COUNT))
        for ind, word in enumerate(cur_sent):
            X_emb[0, ind] = self.__get_word_index(word)
            X_gr[0, ind] = self.index2tags_vector[word.gr_tag]
        return self.model.predict([X_emb, X_gr], verbose=0)[0]


