# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс рифм.

from typing import Tuple

from rupo.main.markup import Word
from rupo.util.preprocess import VOWELS


class Rhymes(object):
    """
    Поиск рифм.
    """

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
        features1 = Rhymes.__get_rhyme_profile(word1)
        features2 = Rhymes.__get_rhyme_profile(word2)
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

    @staticmethod
    def __get_rhyme_profile(word: Word) -> Tuple[int, str, str, str]:
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
