# -*- coding: utf-8 -*-
# Авторы: Гусев Илья
# Описание: Контейнер языковой модели.

import numpy as np
from typing import List


class ModelContainer(object):
    """
    Контейнер языковой модели.
    """
    def get_model(self, word_indices: List[int]) -> np.array:
        """
        Получение проекции языковой модели.
        
        :param word_indices: индексы предыдущих слов.
        :return: вероятности следующего слова.
        """
        raise NotImplementedError()
