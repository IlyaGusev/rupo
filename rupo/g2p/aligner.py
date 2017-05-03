# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Выравнивание слова и транскрпции.


class Aligner:
    russian_map = {
        "а": ["ɐ", "a", "ɑ", "ə", "æ"],
        "б": ["b", "p"],
        "в": ["f", "v"],
        "г": ["g", "ɡ", "ɣ", "v", "x"],
        "д": ["d", "t", "ʦ"],
        "е": ["ɛ", "e", "ə", "ɪ", "ɨ"],
        "ё": ["ɵ", "ɛ", "ʏ", "ɵ", "o"],
        "ж": ["ʂ", "ɕ", "ʐ", "ʑ"],
        "з": ["ɕ", "s", "z", "ʑ"],
        "и": ["i", "ɪ", "y", "ʏ", "ɨ"],
        "й": ["j"],
        "к": ["k"],
        "л": ["ɫ", "l"],
        "м": ["m", "ɱ"],
        "н": ["n", "ɲ"],
        "о": ["ɐ", "o", "ə", "ɔ", "ɵ"],
        "п": ["p"],
        "р": ["r", "ɾ"],
        "с": ["s", "ɕ", "z"],
        "т": ["t", "ʦ"],
        "у": ["u", "ʉ", "ʊ"],
        "ф": ["f"],
        "х": ["ɣ", "x"],
        "ц": ["ʦ", "t"],
        "ч": ["ʂ", "ɕ", "ʧ", "ʨ", "ɕ"],
        "ш": ["ʂ", "ʧ", "ʃ"],
        "щ": ["ɕ", "ʑ"],
        "ь": ["ʲ"],
        "ы": ["ɨ"],
        "ъ": ["j"],
        "э": ["ɛ", "ɪ"],
        "ю": ["ʉ", "ʊ", "u", "ɨ"],
        "я": ["ə", "æ", "ɪ", "a"],
        "-": [],
        " ": []
    }

    @staticmethod
    def align(graphemes, phonemes, sigma=0):
        matrix, trace = Aligner.__build_align_matrix(graphemes, phonemes, sigma)
        g, p = Aligner.__process_align_trace(trace, graphemes, phonemes)
        return g, p

    @staticmethod
    def __build_align_matrix(first_string, second_string, sigma=1):
        f = len(first_string)
        s = len(second_string)
        matrix = [[0 for j in range(s + 1)] for i in range(f + 1)]
        trace = [[0 for j in range(s + 1)] for i in range(f + 1)]

        for i in range(1, f + 1):
            matrix[i][0] = -i * sigma
        for i in range(1, s + 1):
            matrix[0][i] = -i * sigma
        for i in range(1, f + 1):
            for j in range(1, s + 1):
                indel2 = matrix[i - 1][j] - sigma
                indel1 = matrix[i][j - 1] - sigma
                score = 1 if second_string[j - 1] in Aligner.russian_map[first_string[i - 1]] else 0
                replace = matrix[i - 1][j - 1] + score
                scores = [indel2, indel1, replace]
                matrix[i][j] = max(scores)
                trace[i][j] = scores.index(matrix[i][j])
        return matrix, trace

    @staticmethod
    def __process_align_trace(trace, first_string, second_string):
        row = len(first_string)
        col = len(second_string)
        insert_indel = lambda word, pos: word[:pos] + ' ' + word[pos:]
        while row != 0 and col != 0:
            if trace[row][col] == 0:
                second_string = insert_indel(second_string, col)
                row -= 1
            elif trace[row][col] == 1:
                first_string = insert_indel(first_string, row)
                col -= 1
            else:
                row -= 1
                col -= 1
        for i in range(row):
            second_string = insert_indel(second_string, 0)
        for i in range(col):
            first_string = insert_indel(first_string, 0)
        return first_string, second_string
