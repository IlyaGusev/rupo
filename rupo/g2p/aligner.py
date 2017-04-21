# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Выравнивание слова и транскрпции.

from typing import Tuple, Set


class Aligner:
    russian_map = {
        "а": ["ɐ", "a", "ɑ", "ə", "æ"],
        "б": ["b", "p"],
        "в": ["f", "v"],
        "г": ["g", "ɡ", "ɣ", "v", "x"],
        "д": ["d", "t"],
        "е": ["ɛ", "e", "ə", "ɪ", "ɨ", "j", "ʝ"],
        "ё": ["ɵ"],
        "ж": ["ʂ", "ɕ", "ʐ", "ʑ"],
        "з": ["ɕ", "s", "z", "ʑ"],
        "и": ["i"],
        "й": ["j"],
        "к": ["k"],
        "л": ["ɫ"],
        "м": ["m", "ɱ"],
        "н": ["n", "ɲ"],
        "о": ["ɐ", "o", "ə"],
        "п": ["p"],
        "р": ["r", "ɾ"],
        "с": ["s", "ɕ", "z"],
        "т": ["t"],
        "у": ["u", "ʉ", "ʊ"],
        "ф": ["f"],
        "х": ["ɣ", "x"],
        "ц": ["ʦ"],
        "ч": ["ʂ", "ɕ", "ʧ", "ʨ", "ɕ"],
        "ш": ["ʂ", "ʧ"],
        "щ": ["ɕ", "ʑ"],
        "ь": ["ʲ"],
        "ы": ["ɨ"],
        "ъ": [],
        "э": ["ɛ", "ɪ"],
        "ю": ["ʉ", "ʊ", "j", "ʝ"],
        "я": ["ə", "æ", "ɪ", "j", "ʝ"],
        "-": [],
        " ": []
    }

    @staticmethod
    def align_phonemes(graphemes: str, phonemes: str) -> Tuple[str, str]:
        """
        Выравнивание графем и фонем.

        :param graphemes: графемы.
        :param phonemes: фонемы.
        :return: выровненная пара.
        """
        diff = len(graphemes) - len(phonemes)
        phonemes_variants = Aligner.__alignment_variants(phonemes, diff, set()) \
            if diff > 0 else [phonemes]
        graphemes_variants = Aligner.__alignment_variants(graphemes, abs(diff), set()) \
            if diff < 0 else [graphemes]
        scores = {}
        for g in graphemes_variants:
            for p in phonemes_variants:
                assert len(g) == len(p)
                scores[(g, p)] = Aligner.__score_alignment(g, p)
        return max(scores, key=scores.get)

    @staticmethod
    def __alignment_variants(symbols: str, space_count: int, spaces: Set[str]) -> Set[str]:
        """
        Получение вариантов выравнивания.

        :param symbols: буквы.
        :param space_count: количество пробелов, которые осталось расставить
        :param spaces: позиции пробелов.
        :return: варианты выравнивания.
        """
        if space_count == 0:
            answer = ""
            next_symbol = 0
            for i in range(len(symbols) + len(spaces)):
                if i in spaces:
                    answer += " "
                else:
                    answer += symbols[next_symbol]
                    next_symbol += 1
            return {answer}
        variants = set()
        for j in range(len(symbols) + space_count):
            if j not in spaces:
                variants |= Aligner.__alignment_variants(symbols, space_count - 1, spaces | {j})
        return variants

    @staticmethod
    def __score_alignment(graphemes: str, phonemes: str) -> int:
        """
        Оценка выравнивания.

        :param graphemes: графемы.
        :param phonemes: фонемы.
        :return: оценка.
        """
        score = 0
        for i in range(len(graphemes)):
            grapheme = graphemes[i]
            phoneme = phonemes[i]
            if phoneme == " ":
                if i-1 >= 0 and phonemes[i-1] in Aligner.russian_map[grapheme]:
                    score += 0.5
            elif grapheme == " ":
                if i+1 < len(graphemes) and graphemes[i+1] != " " and \
                                phoneme in Aligner.russian_map[graphemes[i+1]]:
                    score += 0.5
            elif phoneme in Aligner.russian_map[grapheme]:
                score += 1
        return score

