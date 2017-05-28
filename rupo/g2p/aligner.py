# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Выравнивание слова и транскрпции.

from typing import List, Tuple, Dict
from collections import defaultdict

from rupo.settings import RU_GRAPHEME_SET
from rupo.g2p.phonemes import Phonemes


class Aligner:
    def __init__(self, grapheme_set: str=RU_GRAPHEME_SET):
        self.grapheme_set = grapheme_set
        self.probability_matrix = None

    def align(self, graphemes: str, phonemes: str):
        """
        Выравнивание графем и фонем.
        
        :param graphemes: графическое слово. 
        :param phonemes: фонетическое слово.
        :return: выровненные слова.
        """
        assert self.probability_matrix is not None
        trace = Aligner.__build_align_matrix(graphemes, phonemes, self.probability_matrix)
        g, p = Aligner.__process_align_trace(trace, graphemes, phonemes)
        return g, p

    def train(self, pairs: List[Tuple[str, str]], n_epochs: int=3):
        """
        Обучение EM-алгоритма над словарём пар.
        
        :param pairs: пары графичесих слов и фонетических слов.
        :param n_epochs: количество итерации обучения.
        """
        phoneme_set = "".join(Phonemes.get_all()).replace(" ", "")
        grapheme_set = self.grapheme_set.replace(" ", "")
        # Сначала задаём равномерное распределение.
        probability_matrix = {g: {p: 1.0/len(phoneme_set) for p in phoneme_set} for g in grapheme_set}
        for _ in range(n_epochs):
            g_p_counts = defaultdict(lambda: defaultdict(lambda: 0.0))
            g_counts = defaultdict(lambda: 0.0)
            # E-шаг.
            for graphemes, phonemes in pairs:
                # Считаем динамику с заданной матрицей весов над матрицей из графем и фонем.
                trace = Aligner.__build_align_matrix(graphemes, phonemes, probability_matrix)
                graphemes, phonemes = Aligner.__process_align_trace(trace, graphemes, phonemes)
                # Увеличиваем счётчики, чтобы потом получить апостериорные вероятности.
                for i in range(len(graphemes)):
                    if graphemes[i] != " " and phonemes[i] != " ":
                        g_p_counts[graphemes[i]][phonemes[i]] += 1
                        g_counts[graphemes[i]] += 1
            # M-шаг. Нормализуем вероятности.
            for g, m in probability_matrix.items():
                for p, prob in m.items():
                    probability_matrix[g][p] = g_p_counts[g][p] / g_counts[g] if g_counts[g] != 0 else 0.0
                    # Заплатка, чтобы ʲ не липла к гласным.
                    if p == "ʲ":
                        probability_matrix[g][p] = 0
        self.probability_matrix = probability_matrix

    @staticmethod
    def __build_align_matrix(first_string: str, second_string: str,
                             probability_matrix: Dict[str, Dict[str, float]], sigma: float=0.0):
        """
        Динамика на матрице g * p.
        
        :param first_string: графемы.
        :param second_string: фонемы
        :param probability_matrix: матрица вероятностей переходов.
        :param sigma: штрафы на пропуски (del).
        :return: путь в матрице, по которому восстаналивается выравнивание.
        """
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
        return trace

    @staticmethod
    def __process_align_trace(trace: List[List[int]], first_string: str, second_string: str):
        """
        Восстановление выравнивания по пути в матрице.
        
        :param trace: путь.
        :param first_string: графемы. 
        :param second_string: фонемы.
        :return: выравненные графемы и фонемы.
        """
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
