# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс рифм.

import pickle
from typing import List, Tuple

from rupo.main.markup import Markup, Word
from rupo.util.preprocess import VOWELS
from rupo.util.vocabulary import Vocabulary


class Rhymes(object):
    """
    Поиск, обработка и хранение рифм.
    """
    def __init__(self):
        self.vocabulary = Vocabulary()

    def add_markup(self, markup: Markup) -> None:
        """
        Добавление слов из разметки в словарь.

        :param markup: разметка.
        """
        for line in markup.lines:
            for word in line.words:
                self.vocabulary.add_word(word)

    def save(self, filename: str) -> None:
        """
        Сохранение состояния данных.

        :param filename: путь к модели.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, filename: str) -> None:
        """
        Загрузка состояния данных.

        :param filename: путь к модели.
        """
        with open(filename, "rb") as f:
            rhymes = pickle.load(f)
            self.__dict__.update(rhymes.__dict__)

    def get_word_rhymes(self, word: Word) -> List[Word]:
        """
        Поиск рифмы для данного слова.

        :param word: слово.
        :return: список рифм.
        """
        rhymes = []
        for i in range(len(self.vocabulary.words)):
            if not Rhymes.is_rhyme(word, self.vocabulary.get_word(i), score_border=5):
                continue
            rhymes.append(self.vocabulary.get_word(i))
        return rhymes

    @staticmethod
    def get_rhyme_profile(word: Word) -> Tuple[int, str, str, str]:
        """
        Получение профиля рифмовки (набора признаков для сопоставления).

        :param word: уже акцентуированное слово (Word).
        :return profile: профиль рифмовки.
        """
        # TODO: Переход на фонетическое слово, больше признаков.
        syllable_number = 0
        accented_syllable = ''
        next_syllable = ''
        next_char = ''
        syllables = list(reversed(word.syllables))
        for i in range(len(syllables)):
            syllable = syllables[i]
            if syllable.accent != -1:
                if i != 0:
                    next_syllable = syllables[i - 1].text
                accented_syllable = syllables[i].text
                if syllable.accent + 1 < len(word.text):
                    next_char = word.text[syllable.accent + 1]
                syllable_number = i
                break
        return syllable_number, accented_syllable, next_syllable, next_char

    @staticmethod
    def is_rhyme(word1: Word, word2: Word, score_border: int=4, syllable_number_border: int=4) -> bool:
        """
        Проверка рифмованности 2 слов.

        :param word1: первое слово для проверки рифмы, уже акцентуированное (Word).
        :param word2: второе слово для проверки рифмы, уже акцентуированное (Word).
        :param score_border: граница определния рифмы, чем выше, тем строже совпадение.
        :param syllable_number_border: ограничение на номер слога с конца, на который падает ударение.
        :return result: является рифмой или нет.
        """
        features1 = Rhymes.get_rhyme_profile(word1)
        features2 = Rhymes.get_rhyme_profile(word2)
        count_equality = 0
        for i in range(len(features1[1])):
            for j in range(i, len(features2[1])):
                if features1[1][i] == features2[1][j]:
                    if features1[1][i] in VOWELS:
                        count_equality += 3
                    else:
                        count_equality += 1
        if features1[2] == features2[2] and features1[2] != '' and features2[2] != '':
            count_equality += 2
        elif features1[3] == features2[3] and features1[3] != '' and features2[3] != '':
            count_equality += 1
        return features1[0] == features2[0] and count_equality >= score_border and \
               features1[0] <= syllable_number_border