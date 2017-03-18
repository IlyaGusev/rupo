# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль создания стихотворений.

import numpy as np
from numpy.random import choice
from typing import List

from rupo.generate.filters import Filter, MetreFilter, RhymeFilter
from rupo.main.phonetics import Phonetics
from rupo.metre.metre_classifier import MetreClassifier, CompilationsSingleton
from rupo.util.vocabulary import Vocabulary
from rupo.accents.dict import AccentDict
from rupo.accents.classifier import MLAccentClassifier


class Generator(object):
    """
    Генератор стихов
    """
    def __init__(self, model_container, vocabulary: Vocabulary):
        """
        :param model_container: модель с методом get_model(list)
        :param vocabulary: словарь с индексами.
        """
        self.model_container = model_container
        self.vocabulary = vocabulary

    def generate_poem(self, metre_schema: str="-+", rhyme_pattern: str="abab", n_syllables: int=8,
                      letters_to_rhymes: dict=None) -> str:
        """
        Генерация стихотворения с выбранными параметрами.

        :param metre_schema: схема метра.
        :param rhyme_pattern: схема рифмы.
        :param n_syllables: количество слогов в строке.
        :param letters_to_rhymes: заданные рифмы.
        :return: стихотворение.
        """
        metre_pattern = ""
        while len(metre_pattern) <= n_syllables:
            metre_pattern += metre_schema
        metre_pattern = metre_pattern[:n_syllables]
        metre_filter = MetreFilter(metre_pattern)

        rhyme_filter = RhymeFilter(rhyme_pattern, letters_to_rhymes)

        prev_word_indices = []
        lines = []
        while rhyme_filter.position >= 0:
            words = self.generate_line(metre_filter, rhyme_filter, prev_word_indices)
            lines.append(" ".join(reversed(words)))
        return "\n".join(reversed(lines)) + "\n"

    def generate_line(self, metre_filter: MetreFilter, rhyme_filter: RhymeFilter,
                      prev_word_indices: List[int]) -> List[str]:
        """
        Генерация одной строки с заданными шаблонами метра и рифмы.

        :param metre_filter: фильтр по метру.
        :param rhyme_filter: фильтр по рифме.
        :param prev_word_indices: индексы предыдущих слов.
        :return: слова строка.
        """
        metre_filter.reset()
        result = []
        word_index = self.generate_word(prev_word_indices, [metre_filter, rhyme_filter])
        prev_word_indices.append(word_index)
        result.append(self.vocabulary.get_word(word_index).text.lower())
        while metre_filter.position >= 0:
            word_index = self.generate_word(prev_word_indices, [metre_filter])
            prev_word_indices.append(word_index)
            result.append(self.vocabulary.get_word(word_index).text.lower())
        return result

    def generate_word(self, prev_word_indices: List[int], filters: List[Filter]) -> int:
        """
        Генерация нового слова на основе предыдущих с учётом фильтров.

        :param prev_word_indices: индексы предыдущих слов.
        :param filters: фильтры модели.
        :return: индекс нового слова.
        """
        model = self.model_container.get_model(prev_word_indices)
        for f in filters:
            model = f.filter_model(model, self.vocabulary)
        if sum(model) == 0:
            model = self.model_container.get_model([])
            for f in filters:
                model = f.filter_model(model, self.vocabulary)
            if sum(model) == 0:
                model = self.model_container.get_model([])
        word_index = Generator.__choose(model)
        word = self.vocabulary.get_word(word_index)
        for f in filters:
            f.pass_word(word)
        return word_index

    def generate_poem_by_line(self, line: str, rhyme_pattern: str,
                              accent_dict: AccentDict, accents_classifier: MLAccentClassifier) -> str:
        """
        Генерация стихотвторения по одной строчке.

        :param accent_dict: словарь ударений.
        :param accents_classifier: классификатор.
        :param line: строчка.
        :param rhyme_pattern: шаблон рифмы.
        :return: стихотворение
        """
        markup, result = MetreClassifier.improve_markup(Phonetics.process_text(line, accent_dict), accents_classifier)
        rhyme_word = markup.lines[0].words[-1]
        count_syllables = sum([len(word.syllables) for word in markup.lines[0].words])
        metre_pattern = CompilationsSingleton.get().get_patterns(result.metre, count_syllables)[0]
        metre_pattern = metre_pattern.lower().replace("s", "+").replace("u", "-")
        letters_to_rhymes = {rhyme_pattern[0]: {rhyme_word}}
        generated = self.generate_poem(metre_pattern, rhyme_pattern, len(metre_pattern), letters_to_rhymes)
        poem = line + "\n" + "\n".join(generated.split("\n")[1:])
        return poem

    @staticmethod
    def __choose(model: np.array):
        """
        Выбор слова из языковой модели.

        :param model: языковая модель.
        :return: слово из модели.
        """
        return choice(range(len(model)), 1, p=model)[0]
