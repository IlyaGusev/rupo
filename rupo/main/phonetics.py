# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль разбивки на слоги, проставления ударений и получения начальной разметки.

from typing import List

from rupo.accents.dict import AccentDict
from rupo.main.markup import Syllable, Word, Markup, Line
from rupo.util.preprocess import count_vowels, get_first_vowel_position, \
    VOWELS, CLOSED_SYLLABLE_CHARS


class Phonetics:
    """
    Класс-механизм для фонетического анализа слов.
    """
    @staticmethod
    def get_word_syllables(word: str) -> List[Syllable]:
        """
        Разделение слова на слоги.

        :param word: слово для разбивки на слоги.
        :return syllables: массив слогов слова.
        """
        syllables = []
        begin = 0
        number = 0
        for i in range(len(word)):
            if word[i] not in VOWELS:
                continue
            if i+1 < len(word)-1 and word[i+1] in CLOSED_SYLLABLE_CHARS:
                if i+2 < len(word)-1 and word[i+2] in "ьЬ":
                    # Если после сонорного согласного идёт мягкий знак, заканчиваем на нём. ("бань-ка")
                    end = i+3
                elif i+2 < len(word)-1 and word[i+2] not in VOWELS and \
                        (word[i+2] not in CLOSED_SYLLABLE_CHARS or word[i+1] == "й"):
                    # Если после сонорного согласного не идёт гласная или другой сонорный согласный,
                    # слог закрывается на этом согласном. ("май-ка")
                    end = i+2
                else:
                    # Несмотря на наличие закрывающего согласного, заканчиваем на гласной.
                    # ("со-ло", "да-нный", "пол-ный")
                    end = i+1
            else:
                # Если после гласной идёт не закрывающая согласная, заканчиваем на гласной. ("ко-гда")
                end = i+1
            syllables.append(Syllable(begin, end, number, word[begin:end]))
            number += 1
            begin = end
        if get_first_vowel_position(word) != -1:
            # Добиваем последний слог до конца слова.
            syllables[-1] = Syllable(syllables[-1].begin, len(word), syllables[-1].number,
                                     word[syllables[-1].begin:len(word)])
        return syllables

    @staticmethod
    def get_word_accents(word: str, accents_dict: AccentDict) -> List[int]:
        """
        Определение ударения в слове по словарю. Возможно несколько вариантов ударения.

        :param word: слово для простановки ударений.
        :param accents_dict: экземпляр обёртки для словаря ударений.
        :return accents: позиции букв, на которые падает ударение.
        """
        accents = []
        if count_vowels(word) == 0:
            # Если гласных нет, то и ударений нет.
            pass
        elif count_vowels(word) == 1:
            # Если одна гласная, то на неё и падает ударение.
            accents.append(get_first_vowel_position(word))
        elif word.find("ё") != -1:
            # Если есть буква "ё", то только на неё может падать ударение.
            accents.append(word.find("ё"))
        else:
            # Проверяем словарь на наличие форм с ударениями.
            accents = accents_dict.get_accents(word)
            if 'е' not in word:
                return accents
            # Находим все возможные варинаты преобразований 'е' в 'ё'.
            positions = [i for i in range(len(word)) if word[i] == 'е']
            beam = [word[:positions[0]]]
            for i in range(len(positions)):
                new_beam = []
                for prefix in beam:
                    n = positions[i+1] if i+1 < len(positions) else len(word)
                    new_beam.append(prefix + 'ё' + word[positions[i]+1:n])
                    new_beam.append(prefix + 'е' + word[positions[i]+1:n])
                    beam = new_beam
            # И проверяем их по словарю.
            for permutation in beam:
                if len(accents_dict.get_accents(permutation)) != 0:
                    yo_pos = permutation.find("ё")
                    if yo_pos != -1:
                        accents.append(yo_pos)
        return accents

    @staticmethod
    def process_text(text: str, accents_dict: AccentDict) -> Markup:
        """
        Получение начального варианта разметки по слогам и ударениям.

        :param text: текст для разметки
        :param accents_dict: экземпляр обёртки для словаря ударений
        :return markup: разметка по слогам и ударениям
        """
        begin_word = -1
        begin_line = 0
        lines = []
        words = []
        # TODO: Нормальная токенизация.
        for i in range(len(text)):
            valid_word_symbol = text[i].isalpha() and i != len(text) - 1
            if valid_word_symbol and begin_word == -1:
                begin_word = i
            if not valid_word_symbol and begin_word != -1:
                # Каждое слово разбиваем на слоги.
                word = Word(begin_word, i, text[begin_word:i], Phonetics.get_word_syllables(text[begin_word:i]))
                # Проставляем ударения.
                accents = Phonetics.get_word_accents(word.text.lower(), accents_dict)
                # Сопоставляем ударения слогам.
                word.set_accents(accents)
                words.append(word)
                begin_word = -1
            if text[i] == "\n":
                # Разбиваем по строкам.
                lines.append(Line(begin_line, i+1, text[begin_line:i], words))
                words = []
                begin_line = i+1
        if begin_line != len(text):
            lines.append(Line(begin_line, len(text), text[begin_line:len(text)], words))
        return Markup(text, lines)

    @staticmethod
    def get_improved_word_accent(word: str, accent_dict: AccentDict, accent_classifier) -> int:
        """
        Получение ударения с учётом классификатора.

        :param word: слово.
        :param accent_dict: словарь ударений.
        :param accent_classifier: классификатор ударений.
        :return: индекс ударения.
        """
        dict_accents = Phonetics.get_word_accents(word, accent_dict)
        if len(dict_accents) == 1:
            return dict_accents[0]
        elif len(dict_accents) == 0:
            clf_accent = accent_classifier.classify_accent(word)
            return clf_accent
        else:
            clf_accent = accent_classifier.classify_accent(word)
            intersection = list(set(dict_accents).intersection({clf_accent}))
            if len(intersection) != 0:
                return intersection[0]
            else:
                return dict_accents[0]
