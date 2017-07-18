# -*- coding: utf-8 -*-
# Авторы: Гусев Илья, Анастасьев Даниил
# Описание: Словоформа.

from enum import IntEnum

class LemmaCase(IntEnum):
    """
    Тип капитализации словоформы
    """
    NORMAL_CASE = 1  # Может писаться как с большой буквы, так и с маленькой
    PROPER_CASE = 2  # Может писаться только с большой буквы
    UPPER_CASE = 3  # Может писаться только всеми большими


class WordForm(object):
    """
    Класс словоформы.
    """
    def __init__(self, lemma: str, gram_vector_index: int, text: str, case: LemmaCase=LemmaCase.NORMAL_CASE):
        """
        :param lemma: лемма словоформы (=начальная форма, нормальная форма).
        :param gram_vector_index: индекс грамматического вектора.
        :param text: вокабула словоформы.
        """
        self.lemma = lemma  # type: str
        self.gram_vector_index = gram_vector_index  # type: int
        self.text = text  # type: str
        self.case = case
        
    def set_case(self, case: LemmaCase) -> None:
        self.case = case
        
    def get_text_with_case(self) -> str:
        if self.case == LemmaCase.NORMAL_CASE:
            return self.text
        if self.case == LemmaCase.PROPER_CASE:
            return self.text.capitalize()
        if self.case == LemmaCase.UPPER_CASE:
            return self.text.upper()
        
    def __repr__(self):
        return "<Lemma = {}; GrTag = {}; WordForm = {}; LemmaCase = {}>".format(self.lemma, 
            self.gram_vector_index, self.text, self.case)

    def __eq__(self, other):
        return (self.lemma, self.gram_vector_index, self.text, self.case) == \
            (other.lemma, other.gram_vector_index, other.text, other.case)

    def __hash__(self):
        return hash((self.lemma, self.gram_vector_index, self.text, self.case))