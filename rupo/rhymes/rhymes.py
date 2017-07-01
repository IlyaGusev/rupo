# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс рифм.

from typing import Tuple
from rupo.main.markup import Word
from rupo.util.preprocess import VOWELS
from rupo.generate.word_form_vocabulary import WordFormVocabulary


class Rhymes(object):
    """
    Поиск рифм.
    """

    @staticmethod
    def is_rhyme(word1: Word, word2: Word, score_border: int=4, syllable_number_border: int=4,
                 word_form_vocabulary: WordFormVocabulary=None) -> bool:
        """
        Проверка рифмованности 2 слов.

        :param word1: первое слово для проверки рифмы, уже акцентуированное (Word).
        :param word2: второе слово для проверки рифмы, уже акцентуированное (Word).
        :param score_border: граница определния рифмы, чем выше, тем строже совпадение.
        :param syllable_number_border: ограничение на номер слога с конца, на который падает ударение.
        :param word_form_vocabulary: словарь словоформ.
        :return result: является рифмой или нет.
        """
        if word_form_vocabulary is not None:
            lemma1 = word_form_vocabulary.get_word_forms_by_text(word1.text.lower())[0].lemma.lower()
            lemma2 = word_form_vocabulary.get_word_forms_by_text(word2.text.lower())[0].lemma.lower()
            if lemma1 == lemma2:
                return False
        features1 = Rhymes.__get_rhyme_profile(word1)
        features2 = Rhymes.__get_rhyme_profile(word2)
        count_equality = 0
        for i, ch1 in enumerate(features1[1]):
            for j in range(i, len(features2[1])):
                ch2 = features2[1][j]
                if ch1 != ch2:
                    continue
                count_equality += 1
                if ch1 in VOWELS:
                    count_equality += 2
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
        stressed_syllable = ''
        next_syllable = ''
        next_char = ''
        syllables = list(reversed(word.syllables))
        for i, syllable in enumerate(syllables):
            if syllable.accent == -1:
                continue
            if i != 0:
                next_syllable = syllables[i - 1].text
            stressed_syllable = syllables[i].text
            if syllable.accent + 1 < len(word.text):
                next_char = word.text[syllable.accent + 1]
            syllable_number = i
            break
        return syllable_number, stressed_syllable, next_syllable, next_char
