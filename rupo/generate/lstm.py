from typing import List
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Activation, Embedding, TimeDistributed, Dense, Dropout, Reshape, Merge, Highway, \
                         LSTM, Convolution2D, MaxPooling2D, BatchNormalization, SpatialDropout1D, Masking
from keras.callbacks import Callback
from keras.models import model_from_json
from rupo.generate import lemmatized_vocabulary

class LSTM_Container(object):
    def __init__(self, path : str):
        json_string = open(path + '.json', 'r', encoding='utf8').readline()
        self.model = model_from_json(json_string)
        self.model.load_weights(path + '_weights.h5')
        self.lemmatized_vocabulary = pickle.load( open(path + "Voc.pkl", "rb") )
        self.num_of_words = len(self.lemmatized_vocabulary.lemmatizedWords)
        self.index2tags_vector = pickle.load( open(path + "_index2tags_vector.pkl", "rb") )
        self.GRAMMEMES_COUNT = 54
        self.SOFTMAX_SIZE = 50000

    def __get_word_index(self, word):
        return min(self.lemmatized_vocabulary.get_word_form_index(word), self.SOFTMAX_SIZE)

    def get_model(self, word_indices: List[int]) -> np.array:
        if len(word_indices) == 0:
            return np.full(self.num_of_words, 1 / self.num_of_words, dtype=np.float)
        cur_sent = [self.lemmatized_vocabulary.get_word_form_by_index(ind) for ind in word_indices]
        X_emb = np.zeros((1, len(cur_sent)))
        X_gr = np.zeros((1, len(cur_sent), self.GRAMMEMES_COUNT))
        for ind, word in enumerate(cur_sent):
            X_emb[0, ind] = self.__get_word_index(word)
            X_gr[0, ind] = self.index2tags_vector[word.gr_tag]
        return self.model.predict([X_emb, X_gr], verbose=0)[0]


