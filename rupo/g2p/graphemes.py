from typing import List

from rupo.main.markup import Syllable
from rupo.util.preprocess import VOWELS, CLOSED_SYLLABLE_CHARS, get_first_vowel_position


class Graphemes:
    @staticmethod
    def get_syllables(word: str) -> List[Syllable]:
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
                part_syllables = Graphemes.get_syllables(part)
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
            if i + 1 < len(word) - 1 and word[i + 1] in CLOSED_SYLLABLE_CHARS:
                if i + 2 < len(word) - 1 and word[i + 2] in "ьЬ":
                    # Если после сонорного согласного идёт мягкий знак, заканчиваем на нём. ("бань-ка")
                    end = i + 3
                elif i + 2 < len(word) - 1 and word[i + 2] not in VOWELS and \
                        (word[i + 2] not in CLOSED_SYLLABLE_CHARS or word[i + 1] == "й"):
                    # Если после сонорного согласного не идёт гласная или другой сонорный согласный,
                    # слог закрывается на этом согласном. ("май-ка")
                    end = i + 2
                else:
                    # Несмотря на наличие закрывающего согласного, заканчиваем на гласной.
                    # ("со-ло", "да-нный", "пол-ный")
                    end = i + 1
            else:
                # Если после гласной идёт не закрывающая согласная, заканчиваем на гласной. ("ко-гда")
                end = i + 1
            syllables.append(Syllable(begin, end, number, word[begin:end]))
            number += 1
            begin = end
        if get_first_vowel_position(word) != -1:
            # Добиваем последний слог до конца слова.
            syllables[-1] = Syllable(syllables[-1].begin, len(word), syllables[-1].number,
                                     word[syllables[-1].begin:len(word)])
        return syllables