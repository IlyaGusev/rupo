import numpy as np

from typing import List
from keras.models import model_from_json
from keras.models import Model, load_model
from keras.layers import Input, Embedding, Dense, Merge, LSTM, SpatialDropout1D, Masking
from keras import backend as K

from rupo.generate.word_form import WordForm
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.generate.grammeme_vectorizer import GrammemeVectorizer
from rupo.settings import GENERATOR_LSTM_MODEL_PATH


class BatchGenerator:
    def __init__(self, filenames, batch_size, softmax_size, sentence_maxlen,
                 word_form_vocabulary: WordFormVocabulary, grammeme_vectorizer: GrammemeVectorizer):
        self.filenames = filenames
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
        for filename in self.filenames:
            with open(filename, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        sentences.append([])
                    else:
                        word, lemma, pos, tags = line.split('\t')[:4]
                        gram_vector_index = self.grammeme_vectorizer.name_to_index[pos+"#"+tags]
                        sentences[-1].append(WordForm(lemma + '_' + pos, gram_vector_index, word))
                    if len(sentences) >= self.batch_size:
                        yield self.__to_tensor(sentences)
                        sentences = [[]]


class LSTMGenerator:
    def __init__(self, softmax_size=60000, external_batch_size=10000, nn_batch_size=768, sentence_maxlen=10):
        self.softmax_size = softmax_size
        self.external_batch_size = external_batch_size
        self.nn_batch_size = nn_batch_size
        self.sentence_maxlen = sentence_maxlen
        self.word_form_vocabulary = None
        self.grammeme_vectorizer = None
        self.model = None

    def prepare(self, filenames):
        self.grammeme_vectorizer = GrammemeVectorizer()
        for filename in filenames:
            self.grammeme_vectorizer.collect_grammemes(filename)
        for filename in filenames:
            self.grammeme_vectorizer.collect_possible_vectors(filename)
        self.grammeme_vectorizer.save()

        self.word_form_vocabulary = WordFormVocabulary()
        for filename in filenames:
            self.word_form_vocabulary.load_from_corpus(filename, self.grammeme_vectorizer)
        self.word_form_vocabulary.sort()
        self.word_form_vocabulary.save()
        self.word_form_vocabulary.inflate_vocab(self.softmax_size)

    def load(self, model_filename):
        self.model = load_model(model_filename)

    def load_with_weights(self, json_filename, weights_filename):
        json_string = open(json_filename, 'r', encoding='utf8').readline()
        self.model = model_from_json(json_string)
        self.model.load_weights(weights_filename)

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

    def train(self, filenames):
        batch_generator = BatchGenerator(filenames,
                                         batch_size=self.external_batch_size,
                                         softmax_size=self.softmax_size,
                                         sentence_maxlen=self.sentence_maxlen,
                                         word_form_vocabulary=self.word_form_vocabulary,
                                         grammeme_vectorizer=self.grammeme_vectorizer)
        for big_epoch in range(1000):
            print('------------Big Epoch {}------------'.format(big_epoch))
            for epoch, (X1, X2, y) in enumerate(batch_generator):
                self.model.fit([X1, X2], y, batch_size=self.nn_batch_size, epochs=1, verbose=1)
                indices = [np.random.randint(0, self.softmax_size)]
                for i in range(10):
                    indices.append(self._sample(self.predict(indices)))
                sentence = [self.word_form_vocabulary.get_word_form_by_index(index) for index in indices]
                print('Sentence', end=': ')
                for word in sentence[::-1]:
                    print(word.text, end=' ')
                print()

                if epoch != 0 and epoch % 10 == 0:
                    self.model.save(GENERATOR_LSTM_MODEL_PATH)

    def predict(self, word_indices: List[int]):
        if len(word_indices) == 0:
            return np.full(self.softmax_size, 1.0/self.softmax_size, dtype=np.float)
        cur_sent = [self.word_form_vocabulary.get_word_form_by_index(ind) for ind in word_indices]
        x_emb = np.zeros((1, len(cur_sent)))
        x_gram = np.zeros((1, len(cur_sent), self.grammeme_vectorizer.grammemes_count()))
        for index, word in enumerate(cur_sent):
            x_emb[0, index] = self.word_form_vocabulary.get_word_form_index_min(word, self.softmax_size)
            x_gram[0, index] = self.grammeme_vectorizer.vectors[word.gram_vector_index]
        return self.model.predict([x_emb, x_gram], verbose=0)[0]

    @staticmethod
    def _sample(prob, temperature=1.0):
        prob = prob[:-1]
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.random.choice(len(prob), p=prob)


class LSTMModelContainer(object):
    def __init__(self):
        self.lstm = LSTMGenerator()
        self.lstm.load(GENERATOR_LSTM_MODEL_PATH)

    def get_model(self, word_indices: List[int]) -> np.array:
        return self.lstm.predict(word_indices)


