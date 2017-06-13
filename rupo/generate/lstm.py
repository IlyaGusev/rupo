import pickle
import numpy as np
import re
from typing import List
from contextlib import redirect_stdout
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input, Activation, Embedding, Dense, Merge, LSTM, SpatialDropout1D, Masking
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from rupo.settings import GENERATOR_MODEL_DESCRIPTION, GENERATOR_MODEL_WEIGHTS, GENERATOR_LEMMA_VOCAB_PATH, \
    GENERATOR_TAGS_VECTORS

def hard_tanh(x):
    one = K.constant(1)
    neg_one = K.constant(-1)
    return K.minimum(K.maximum(x, neg_one), one)
get_custom_objects().update({'custom_activation': Activation(hard_tanh)})

GRAMMEMES = []
def binarize(i, length):
    vector = np.zeros(length + 1, dtype=np.int)
    vector[i] = 1
    return list(vector)

with open('ClassesNames.txt', encoding='utf8') as f:
    for line in f:
        fields = line.strip().split(' ')
        GRAMMEMES.append({x: binarize(i, len(fields)) for i, x in enumerate(fields)})
        GRAMMEMES[-1][u'UNK'] = binarize(len(fields), len(fields))
GRAMMEMES_COUNT = sum(len(x) for x in GRAMMEMES)

def convert_tags(pos, tags):
    tags = tags.split('|')
    tags_vector = GRAMMEMES[0][pos] if pos in GRAMMEMES[0] else GRAMMEMES[0][u'UNK']
    tags_vector = tags_vector[:]
    for mapping in GRAMMEMES[1:]:
        if any(tag in mapping for tag in tags):
            for tag in tags:
                if tag in mapping:
                    tags_vector.extend(mapping[tag])
        else:
            tags_vector.extend(mapping[u'UNK'])
    return tags_vector


classesToValue = {}
with open('classesList.txt', encoding='utf-8') as f:
    for line in f:
        line = line[:-2]
        pos, val = line[:line.index('\t')], line[line.index('\t') + 1:]
        classesToValue[int(pos)] = val.replace('\t', ' ')

classesToValue[-1] = "Unknown"

with open('classesMapping.txt', encoding='utf-8') as f:
    classesMapping = {int(line[:-1].split('\t')[1]): int(line.split('\t')[0]) for line in f}

classesMappingRev = {classesMapping[x]: x for x in classesMapping}

space = re.compile('\\s+')

def format_gram_value(gram_val):
    first_tab = gram_val.find(' ')
    if first_tab != -1:
        pos, grval = gram_val[: first_tab], gram_val[first_tab + 1:]
        grval = grval.replace('-', ' ')
        grval = space.subn(' ', grval)[0].strip()
        grval = grval.replace(' ', '|')
        if len(grval) != 0:
            tags = {a.split('=')[0]: a.split('=')[1] if a != '_' else '_' for a in grval.split('|')}
            if pos == 'VERB' and 'Tense' not in tags \
                    and (tags["VerbForm"] == "Fin" and tags["Mood"] == "Ind"
                         or tags["VerbForm"] == "Conv"):
                tags["Tense"] = "Past"
            grval = ''
            for tag in tags:
                grval += tag + '=' + tags[tag] + '|'
        else:
            grval = '_ '
        return pos + '\t' + grval[:-1]
    else:
        return gram_val + '\t_'


index2tags_vector = {}
for i in range(286 + 1):
    if i >= 2:
        pos, tags = format_gram_value(classesToValue[classesMapping[i] - 2]).split('\t')
        index2tags_vector[i] = convert_tags(pos, tags)
    elif i == 0:
        index2tags_vector[i] = list(np.zeros(GRAMMEMES_COUNT, dtype=np.int))
    elif i == 1:
        index2tags_vector[i] = convert_tags('', '')

class BatchGenerator():
    def __init__(self, fname, batch_size):
        self.fname = fname
        self.batch_size = batch_size

    @staticmethod
    def __generate_seqs(sents):
        seqs, next_words = [], []
        for sent in sents:
            sent = sent[::-1]
            for i in range(1, len(sent)):
                if lemmatizedVocabulary.get_word_form_index(sent[i]) >= SOFTMAX_SIZE:
                    continue
                seqs.append(sent[max(0, i - SENT_LEN): i])
                next_words.append(sent[i])
        return seqs, next_words

    @staticmethod
    def __to_tensor(sents):
        sents, next_words = BatchGenerator.__generate_seqs(sents)
        max_len = max(len(sent) for sent in sents)
        X_emb = np.zeros((len(sents), max_len), dtype=np.int)
        X_grammemes = np.zeros((len(sents), max_len, GRAMMEMES_COUNT), dtype=np.int)
        y = np.zeros(len(sents), dtype=np.int)
        for i in range(len(sents)):
            X_emb[i, -len(sents[i]):] = [get_word_index(x) for x in sents[i]]
            X_grammemes[i, -len(sents[i]):] = [index2tags_vector[x.gr_tag] for x in sents[i]]
            y[i] = get_word_index(next_words[i])
        return X_emb, X_grammemes, y

    @staticmethod
    def to_tensor(words_indices):
        return BatchGenerator.__to_tensor([idx2word[ind] for ind in words_indices])

    def __iter__(self):
        sents = [[]]
        with open(self.fname, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    sents.append([])
                else:
                    word, lemma, pos, tags, index = line.split('\t')
                    sents[-1].append(LemmatizedWord(lemma + '_' + pos, int(index), word))
                if len(sents) >= self.batch_size:
                    yield BatchGenerator.__to_tensor(sents)
                    sents = [[]]


class EvalCallback(Callback):
    def __init__(self, name):
        self._name = name

    def write(self, message):
        print('[{}] {}'.format(strftime("%H:%M:%S", localtime()), message))

    def _sample(self, prob, temperature=1.0):
        prob = prob[:-1]  # Не хотим предсказывать UNKNOWN_WORD
        prob = np.log(prob) / temperature
        prob = np.exp(prob) / np.sum(np.exp(prob))
        return np.random.choice(len(prob), p=prob)

    def _generate(self):
        cur_sent = [lemmatizedVocabulary.get_word_form_by_index(np.random.randint(0, SOFTMAX_SIZE))]
        for i in range(10):
            X_emb = np.zeros((1, len(cur_sent)))
            X_gr = np.zeros((1, len(cur_sent), GRAMMEMES_COUNT))
            for ind, word in enumerate(cur_sent):
                X_emb[0, ind] = get_word_index(word)
                X_gr[0, ind] = index2tags_vector[word.gr_tag]

            preds = model.predict([X_emb, X_gr], verbose=0)[0]
            cur_sent.append(lemmatizedVocabulary.get_word_form_by_index(self._sample(preds)))

        print('Sentence', end=': ')
        for word in cur_sent[::-1]:
            print(word.word_form, end=' ')
        print()

    def on_epoch_end(self, epoch, logs={}):
        self._generate()

class LSTMGenerator:
    def __init__(self, grammems_count=, softmax_size=60000):
        self.softmax_size = softmax_size

        self.model = None

    def build(self):
        words = Input(shape=(None,), name='words')

        words_embedding = SpatialDropout1D(0.3)(Embedding(len(lemma2index) + 1, 150, name='embeddings')(words))
        grammemes_input = Input(shape=(None, GRAMMEMES_COUNT), name='grammemes')
        grammemes_layer = Masking(mask_value=0.)(grammemes_input)
        grammemes_layer = Dense(25, activation=Activation(hard_tanh))(grammemes_layer)
        grammemes_layer = Dense(25, activation=Activation(hard_tanh))(grammemes_layer)
        layer = Merge(mode='concat', name='LSTM_input')([words_embedding, grammemes_layer])
        layer = LSTM(368, dropout=.2, recurrent_dropout=.2, return_sequences=True, name='LSTM_1')(layer)
        layer = LSTM(368, dropout=.2, recurrent_dropout=.2, return_sequences=False, name='LSTM_2')(layer)

        output = Dense(self.softmax_size + 1, activation='softmax')(layer)

        self.model = Model(inputs=[words, grammemes_input], outputs=[output])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    def train(self):
        batch_generator = BatchGenerator('Data/Poetry_preds.txt_lemmatized_train', 10000)
        name = 'Lemmatized_hard_tanh'
        with open(name + '_log.txt', 'a') as f:
            with redirect_stdout(f):
                callback = EvalCallback(name)
                for big_epoch in range(1000):
                    print('------------Big Epoch {}------------'.format(big_epoch))
                    for epoch, (X1, X2, y) in enumerate(batch_generator):
                        self.model.fit([X1, X2], y, batch_size=768, epochs=1, verbose=2, callbacks=[callback])
                        if epoch != 0 and epoch % 10 == 0:
                            self.model.save_weights(name + '_model.h5')
                        f.flush()


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


