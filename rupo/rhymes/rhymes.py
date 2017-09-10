# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс рифм.

from rupo.util.preprocess import VOWELS
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.stress.word import StressedWord


class RhymeProfile:
    def __init__(self, syllable_count: int, stressed_syllable_number: int,
                 stressed_syllable_text: str, next_syllable_text: str, next_char: str):
        self.syllable_count = syllable_count
        self.stressed_syllable_number = stressed_syllable_number
        self.stressed_syllable_text = stressed_syllable_text
        self.next_syllable_text = next_syllable_text
        self.next_char = next_char

    def __str__(self):
        return "Syllable count: {}; Stressed syllable: {}; " \
               "Stressed syllable text: {}; Next syllable: {}; " \
               "Next char: {}".format(self.syllable_count, self.stressed_syllable_number,
                                      self.stressed_syllable_text, self.next_syllable_text, self.next_char)

    def __repr__(self):
        return self.__str__()


class Rhymes(object):
    """
    Поиск рифм.
    """

    @staticmethod
    def is_rhyme(word1: StressedWord, word2: StressedWord, score_border: int=4, syllable_number_border: int=4,
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
            lemma1 = word_form_vocabulary.get_word_form_by_text(word1.text.lower()).lemma.lower()
            lemma2 = word_form_vocabulary.get_word_form_by_text(word2.text.lower()).lemma.lower()
            if lemma1 == lemma2:
                return False
        profile1 = Rhymes.__get_rhyme_profile(word1)
        profile2 = Rhymes.__get_rhyme_profile(word2)
        score = 0
        for i, ch1 in enumerate(profile1.stressed_syllable_text):
            for j, ch2 in enumerate(profile2.stressed_syllable_text[i:]):
                if ch1 != ch2:
                    continue
                if ch1 in VOWELS:
                    score += 3
                else:
                    score += 1
        if profile1.next_syllable_text == profile2.next_syllable_text and profile1.next_syllable_text != '':
            score += 3
        elif profile1.next_char == profile2.next_char and profile1.next_char != '':
            score += 1
        return (profile1.stressed_syllable_number == profile2.stressed_syllable_number and
                profile1.syllable_count == profile2.syllable_count and
                profile1.stressed_syllable_number <= syllable_number_border and
                score >= score_border)

    @staticmethod
    def __get_rhyme_profile(word: StressedWord) -> 'RhymeProfile':
        """
        Получение профиля рифмовки (набора признаков для сопоставления).

        :param word: уже акцентуированное слово (Word).
        :return profile: профиль рифмовки.
        """
        # TODO: Переход на фонетическое слово, больше признаков.

        profile = RhymeProfile(syllable_count=0,
                               stressed_syllable_number=-1,
                               stressed_syllable_text="",
                               next_syllable_text="",
                               next_char="")
        syllables = list(word.syllables)
        profile.syllable_count = len(syllables)
        for i, syllable in enumerate(reversed(syllables)):
            if syllable.stress == -1:
                continue
            profile.stressed_syllable_text = syllable.text
            profile.stressed_syllable_number = -i-1
            if i != 0:
                profile.next_syllable = syllables[-i].text
            if syllable.stress + 1 < len(word.text):
                profile.next_char = word.text[syllable.stress + 1]
            break
        return profile
