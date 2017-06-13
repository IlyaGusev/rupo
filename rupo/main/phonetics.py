# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль разбивки на слоги, проставления ударений и получения начальной разметки.

from typing import List

from rupo.stress.dict import StressDict
from rupo.main.markup import Syllable, Word, Markup, Line
from rupo.main.tokenizer import Tokenizer, Token
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

        # В случае наличия дефиса разбиваем слова на подслова, находим слоги в них, объединяем.
        if "-" in word:
            word_parts = word.split("-")
            word_syllables = []
            last_part_end = 0
            for part in word_parts:
                part_syllables = Phonetics.get_word_syllables(part)
                if len(part_syllables) == 0:
                    continue
                for i in range(len(part_syllables)):
                    part_syllables[i].begin += last_part_end
                    part_syllables[i].end += last_part_end
                    part_syllables[i].number += len(word_syllables)
                word_syllables += part_syllables
                last_part_end = part_syllables[-1].end + 1
            return word_syllables

        # Для слов или подслов, в которых нет дефиса.
        for i, ch in enumerate(word):
            if ch not in VOWELS:
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
    def get_word_stresses(word: str, stress_dict: StressDict) -> List[int]:
        """
        Определение ударения в слове по словарю. Возможно несколько вариантов ударения.

        :param word: слово для простановки ударений.
        :param stress_dict: экземпляр обёртки для словаря ударений.
        :return stresses: позиции букв, на которые падает ударение.
        """
        stresses = []
        if count_vowels(word) == 0:
            # Если гласных нет, то и ударений нет.
            pass
        elif count_vowels(word) == 1:
            # Если одна гласная, то на неё и падает ударение.
            stresses.append(get_first_vowel_position(word))
        elif word.find("ё") != -1:
            # Если есть буква "ё", то только на неё может падать ударение.
            stresses.append(word.find("ё"))
        else:
            # Проверяем словарь на наличие форм с ударениями.
            stresses = stress_dict.get_stresses(word)
            if 'е' not in word:
                return stresses
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
                if len(stress_dict.get_stresses(permutation)) != 0:
                    yo_pos = permutation.find("ё")
                    if yo_pos != -1:
                        stresses.append(yo_pos)
        return stresses

    @staticmethod
    def process_text(text: str, g2p_model, stress_model, aligner) -> Markup:
        """
        Получение начального варианта разметки по слогам и ударениям.

        :param text: текст для разметки
        :return markup: разметка по слогам и ударениям
        """
        begin_line = 0
        lines = []
        words = []
        text_lines = text.split("\n")
        for text_line in text_lines:
            tokens = [token for token in Tokenizer.tokenize(text_line) if token.token_type == Token.TokenType.WORD]
            for token in tokens:
                word = Word(begin_line + token.begin, begin_line + token.end, token.text,
                            Phonetics.get_word_syllables(token.text))
                # Проставляем ударения.
                stresses = Phonetics.get_g2p_stresses(token.text, g2p_model, stress_model, aligner)
                # Сопоставляем ударения слогам.
                word.set_stresses(stresses)
                words.append(word)
            end_line = begin_line + len(text_line)
            lines.append(Line(begin_line, end_line, text_line, words))
            words = []
            begin_line = end_line + 1
        return Markup(text, lines)

    @staticmethod
    def get_improved_word_stress(word: str, stress_dict: StressDict, stress_classifier) -> int:
        """
        Получение ударения с учётом классификатора.

        :param word: слово.
        :param stress_dict: словарь ударений.
        :param stress_classifier: классификатор ударений.
        :return: индекс ударения.
        """
        dict_stresses = Phonetics.get_word_stresses(word, stress_dict)
        if len(dict_stresses) == 1:
            return dict_stresses[0]
        elif len(dict_stresses) == 0:
            clf_stresses = stress_classifier.classify_stress(word)
            return clf_stresses
        else:
            clf_stresses = stress_classifier.classify_stress(word)
            intersection = list(set(dict_stresses).intersection({clf_stresses}))
            if len(intersection) != 0:
                return intersection[0]
            else:
                return dict_stresses[0]

    @staticmethod
    def get_g2p_stresses(word: str, g2p_model, stress_model, aligner):
        word = word.lower()
        phonemes = g2p_model.predict([word])[0].replace(" ", "")
        stresses = stress_model.predict([phonemes])[0]
        stresses = [i for i, stress in enumerate(stresses) if stress == 1 or stress == 2]
        g, p = aligner.align(word, phonemes)
        stresses = aligner.align_stresses(g, p, stresses, is_grapheme=False)
        for i, stress in enumerate(stresses):
            stresses[i] -= len([ch for ch in g[:stress] if ch == " "])
        stresses = [i for i in stresses if i < len(word)]
        return stresses
