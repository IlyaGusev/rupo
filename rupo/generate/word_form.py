# -*- coding: utf-8 -*-
# Авторы: Гусев Илья, Анастасьев Даниил
# Описание: Словоформа.


class WordForm(object):
    """
    Класс словоформы.
    """
    def __init__(self, lemma: str, gram_vector_index: int, text: str):
        """
        :param lemma: лемма словоформы (=начальная форма, нормальная форма).
        :param gram_vector_index: индекс грамматического вектора.
        :param text: вокабула словоформы.
        """
        self.lemma = lemma  # type: str
        self.gram_vector_index = gram_vector_index  # type: int
        self.text = text  # type: str

    def __repr__(self):
        return "<Lemma = {}; GrTag = {}; WordForm = {}>".format(self.lemma, self.gram_vector_index, self.text)

    def __eq__(self, other):
        return (self.lemma, self.gram_vector_index, self.text) == (other.lemma, other.gram_vector_index, other.text)

    def __hash__(self):
        return hash((self.lemma, self.gram_vector_index, self.text))