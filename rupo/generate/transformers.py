# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль трансформеров языковой модели.

import numpy as np

from rupo.generate.word_form_vocabulary import WordFormVocabulary
from gensim.models import KeyedVectors


class Transformer(object):
    """
    Трансоформер языковой модели.
    """
    def get_coefficients(self) -> np.array:
        raise NotImplementedError()

    def transform_model(self, model: np.array) -> np.array:
        return np.multiply(model, self.get_coefficients())


class DistributionalSemanticTransformer(Transformer):
    """
    Трансоформер на основе семантических векторов.
    """
    def __init__(self, vocabulary: WordFormVocabulary, key_word: str, vectors_filename: str):
        w2v = KeyedVectors.load_word2vec_format(vectors_filename, binary=True)
        coefficients = []
        for i in range(len(vocabulary.word_forms)):
            current_word = vocabulary.get_word_form_by_index(i).lemma
            if key_word in w2v.vocab and current_word in w2v.vocab:
                coefficients.append(1.0 + w2v.similarity(key_word, current_word)*10)
            else:
                coefficients.append(1.0)
        self.coefficients = np.array(coefficients)
        pass

    def get_coefficients(self):
        return self.coefficients

