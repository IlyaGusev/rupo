# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Выравнивание слова и транскрпции.

from rupo.settings import RU_GRAPHEME_SET, RU_G2P_DICT_PATH
from rupo.g2p.phonemes import Phonemes
from collections import defaultdict


class Aligner:
    def __init__(self, grapheme_set=RU_GRAPHEME_SET):
        self.grapheme_set = grapheme_set
        self.probability_matrix = None

    def align(self, graphemes, phonemes, sigma=0):
        assert self.probability_matrix is not None
        matrix, trace = Aligner.__build_align_matrix(graphemes, phonemes, self.probability_matrix, sigma)
        g, p = Aligner.__process_align_trace(trace, graphemes, phonemes)
        return g, p

    def train(self, pairs, sigma=0):
        phoneme_set = "".join(Phonemes.get_all()).replace(" ", "")
        grapheme_set = self.grapheme_set.replace(" ", "")
        probability_matrix = {g: {p: 1.0/len(phoneme_set) for p in phoneme_set} for g in grapheme_set}
        for _ in range(3):
            g_p_counts = defaultdict(lambda: defaultdict(lambda: 0.0))
            g_counts = defaultdict(lambda: 0.0)
            for graphemes, phonemes in pairs:
                matrix, trace = Aligner.__build_align_matrix(graphemes, phonemes, probability_matrix, sigma)
                graphemes, phonemes = Aligner.__process_align_trace(trace, graphemes, phonemes)
                for i in range(len(graphemes)):
                    if graphemes[i] != " " and phonemes[i] != " ":
                        g_p_counts[graphemes[i]][phonemes[i]] += 1
                        g_counts[graphemes[i]] += 1
            for g, m in probability_matrix.items():
                for p, prob in m.items():
                    probability_matrix[g][p] = g_p_counts[g][p] / g_counts[g] if g_counts[g] != 0 else 0.0
                    if p == "ʲ":
                        probability_matrix[g][p] = 0
        self.probability_matrix = probability_matrix

    @staticmethod
    def __build_align_matrix(first_string, second_string, probability_matrix, sigma=0):
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
                score1 = probability_matrix[first_string[i-1]][second_string[j-1]]
                replace1 = matrix[i - 1][j - 1] + score1
                scores = [indel2, indel1, replace1]
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

if __name__ == '__main__':
    with open(RU_G2P_DICT_PATH, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        pairs = [line.strip().split("\t") for line in lines]
        aligner = Aligner()
        aligner.train(pairs)
        for g, p in pairs[::50]:
            print(aligner.align(g, p))