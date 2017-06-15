# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль фильтров языковой модели по разным признакам.

from collections import defaultdict
from typing import List

import numpy as np

from rupo.main.markup import Word
from rupo.main.vocabulary import Vocabulary
from rupo.rhymes.rhymes import Rhymes
from rupo.generate.word_form_vocabulary import WordFormVocabulary


class Filter(object):
    """
    Фильтр языковой модели.
    """
    def filter_word(self, word: Word) -> bool:
        raise NotImplementedError()

    def pass_word(self, word: Word) -> None:
        raise NotImplementedError()

    def revert_word(self, word: Word) -> None:
        raise NotImplementedError()

    def is_completed(self) -> bool:
        raise NotImplementedError()

    def filter_model(self, model: np.array, vocabulary: Vocabulary) -> np.array:
        """
        Фильтрация языковой модели.

        :param model: изначальная модель.
        :param vocabulary: словарь
        :return: модель после фильтрации и нормирования.
        """
        for i in range(len(model)):
            if not self.filter_word(vocabulary.get_word(i)):
                model[i] = 0.0
        # if np.sum(model) != 0:
        #     model /= np.sum(model)
        return model

    def filter_words(self, words: List[Word]) -> List[Word]:
        """
        Фильтрация набора слов.

        :param words: слова.
        :return: слова после фильтрации.
        """
        return [word for word in words if self.filter_word(word)]


class MetreFilter(Filter):
    """
    Фильтр по шаблону метра.
    """
    def __init__(self, metre_pattern: str):
        self.metre_pattern = metre_pattern
        self.position = len(metre_pattern) - 1

    def filter_word(self, word: Word) -> bool:
        """
        Фильтрация слова по метру в текущей позиции.

        :param word: слово.
        :return: подходит слово или нет.
        """
        syllables_count = len(word.syllables)
        if syllables_count == 0:
            return False
        if syllables_count > self.position + 1:
            return False
        for i in range(syllables_count):
            syllable = word.syllables[i]
            syllable_number = self.position - syllables_count + i + 1
            if syllables_count >= 2 and syllable.accent == -1 and self.metre_pattern[syllable_number] == "+":
                for j in range(syllables_count):
                    other_syllable = word.syllables[j]
                    other_syllable_number = other_syllable.number - syllable.number + syllable_number
                    if i != j and other_syllable.accent != -1 and self.metre_pattern[other_syllable_number] == "-":
                        return False
        return True

    def pass_word(self, word: Word) -> None:
        """
        Сдвинуть позицию в шаблоне метра на слово.

        :param word: слово.
        """
        self.position -= len(word.syllables)

    def revert_word(self, word: Word) -> None:
        """
        Сдвинуть позицию в шаблоне метра на слово назад.

        :param word: слово.
        """
        self.position += len(word.syllables)

    def reset(self) -> None:
        """
        Сброс позиции в шаблоне.
        """
        self.position = len(self.metre_pattern) - 1

    def is_completed(self):
        """
        :return: закончена ли генерация по фильтру?
        """
        return self.position < 0


class RhymeFilter(Filter):
    """
    Фильтр по шаблону рифмы.
    """
    def __init__(self, rhyme_pattern: str, letters_to_rhymes: dict=None,
                 word_form_vocabulary: WordFormVocabulary=None, score_border=4):
        self.word_form_vocabulary = word_form_vocabulary  # type: WordFormVocabulary
        self.rhyme_pattern = rhyme_pattern
        self.position = len(self.rhyme_pattern) - 1
        self.letters_to_rhymes = defaultdict(set)
        self.score_border = score_border
        if letters_to_rhymes is not None:
            for letter, words in letters_to_rhymes.items():
                for word in words:
                    self.letters_to_rhymes[letter].add(word)

    def filter_word(self, word: Word) -> bool:
        """
        Фильтрация слова по рифме в текущей позиции.

        :param word: слово.
        :return: подходит слово или нет.
        """
        if len(word.syllables) <= 1:
            return False
        if len(self.letters_to_rhymes[self.rhyme_pattern[self.position]]) == 0:
            return True
        first_word = list(self.letters_to_rhymes[self.rhyme_pattern[self.position]])[0]

        is_rhyme = Rhymes.is_rhyme(first_word, word, score_border=self.score_border, syllable_number_border=2,
                                   word_form_vocabulary=self.word_form_vocabulary) and \
            first_word.text != word.text
        return is_rhyme

    def pass_word(self, word: Word) -> None:
        """
        Сдвинуть позицию в шаблоне рифмы на строчку.

        :param word: рифмующееся слово.
        """
        self.letters_to_rhymes[self.rhyme_pattern[self.position]].add(word)
        self.position -= 1

    def revert_word(self, word: Word) -> None:
        """
        Сдвинуть позицию в шаблоне рифмы на строчку назад.

        :param word: рифмующееся слово.
        """
        self.position += 1
        self.letters_to_rhymes[self.rhyme_pattern[self.position]].remove(word)

    def is_completed(self):
        """
        :return: закончена ли генерация по фильтру?
        """
        return self.position < 0

    def reset(self) -> None:
        """
        Сброс позиции в шаблоне.
        """
        self.position = len(self.rhyme_pattern) - 1
