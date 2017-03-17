import xml.etree.ElementTree as etree
from collections import Counter

import numpy as np
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.models import Sequential
from keras.optimizers import RMSprop

from rupo.main.markup import Markup as Markup
from rupo.util.vocabulary import Vocabulary

UNKNOWN_WORD = "#########"


class LSTMModelContainer:
    def __init__(self, markup_dump_filename: str):
        self.nn_model = None
        self.vocabulary = Vocabulary()
        self.get_vocabulary(markup_dump_filename, 60000)
        self.train(60000, 10, )

    def get_model(self, word_indices, sentence_length):
        x = np.zeros((1, sentence_length, len(self.vocabulary.words)))
        for t, index in enumerate(word_indices):
            x[0, t, index] = 1.

        model = self.nn_model.predict(x, verbose=0)[0]
        return model

    def get_vocabulary(self, markup_dump_filename, words_count):
        words_counter = Counter()
        for event, elem in etree.iterparse(markup_dump_filename, events=['end']):
            if event == 'end' and elem.tag == 'markup':
                markup = Markup()
                markup.from_xml(etree.tostring(elem, encoding='utf-8', method='xml'))
                for line in markup.lines:
                    for word in line.words:
                        self.vocabulary.add_word(word)
                        words_counter[word.get_short()] += 1
        self.vocabulary.shrink([i[0] for i in words_counter.most_common(words_count)])
        self.vocabulary.add_word(UNKNOWN_WORD)
        self.vocabulary.add_word("\n")

    def train(self, markup_dump_filename, sentence_length=10, step=3, local_batch_size=128,
              global_batch_size=8192, iterations_count=30):
        model = self.__build_model(sentence_length, local_batch_size)
        for iteration in range(iterations_count):
            text = []
            text_index = 0
            for event, elem in etree.iterparse(markup_dump_filename, events=['end']):
                if event == 'end' and elem.tag == 'markup':
                    markup = Markup()
                    markup.from_xml(etree.tostring(elem, encoding='utf-8', method='xml'))
                    markup.text = markup.text.replace("\\n", "\n")
                    for line in markup.lines:
                        for word in line.words:
                            text.append(word.get_short())
                        text.append("\n")
                    if len(text) >= global_batch_size:
                        x, y = self.__get_batch(text, global_batch_size, sentence_length, step)
                        model.fit(x, y, batch_size=local_batch_size, nb_epoch=1)
                        text[:] = []
                    text_index += 1
                    if text_index % 1000 == 0:
                        print(text_index)

    def __build_model(self, sentence_length, local_batch_size):
        words_count = len(self.vocabulary.words)
        model = Sequential()
        model.add(LSTM(local_batch_size, input_shape=(sentence_length, words_count), stateful=False))
        model.add(Dense(words_count))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.005))
        return model

    def __get_batch(self, text, size, sentence_length, step):
        words_count = len(self.vocabulary.words)
        sentences = []
        next_words = []
        end = size
        if end > len(text) - sentence_length:
            end = len(text) - sentence_length
        for i in range(0, end, step):
            sentences.append(text[i: i + sentence_length])
            next_words.append(text[i + sentence_length])
        X = np.zeros((len(sentences), sentence_length, words_count), dtype=np.bool)
        y = np.zeros((len(sentences), words_count), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                if word in self.vocabulary.shorts_set:
                    X[i, t, self.vocabulary.word_to_index[word]] = 1
                else:
                    X[i, t, self.vocabulary.word_to_index[UNKNOWN_WORD]] = 1
            if next_words[i] in self.vocabulary.shorts_set:
                y[i, self.vocabulary.word_to_index[next_words[i]]] = 1
            else:
                y[i, self.vocabulary.word_to_index[UNKNOWN_WORD]] = 1
        return X, y
