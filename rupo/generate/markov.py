# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль марковских цепей.

import os
import pickle
import sys
from collections import Counter
from typing import List

import numpy as np

from rupo.main.markup import Markup
from rupo.util.vocabulary import Vocabulary
from rupo.files.reader import Reader, FileTypeEnum


class MarkovModelContainer(object):
    """
    Марковские цепи.
    """
    def __init__(self, dump_filename: str, markup_dump_path: str=None):
        self.transitions = list()
        self.vocabulary = Vocabulary()

        # Делаем дамп модели для ускорения загрузки.
        if os.path.isfile(dump_filename):
            with open(dump_filename, "rb") as f:
                markov = pickle.load(f)
                self.__dict__.update(markov.__dict__)
        else:
            sys.stdout.write("Starting\n")
            sys.stdout.flush()
            i = 0
            markups = Reader.read_markups(markup_dump_path, FileTypeEnum.XML, is_processed=True)
            for markup in markups:
                self.add_markup(markup)
                i += 1
                if i % 500 == 0:
                    sys.stdout.write(str(i)+"\n")
                    sys.stdout.flush()

            with open(dump_filename, "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def generate_chain(self, words: List[int]) -> List[Counter]:
        """
        Генерация переходов в марковских цепях с учётом частотности.

        :param words: вершины цепи.
        :return: обновленные переходы.
        """
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i+1]
            self.transitions[current_word][next_word] += 1
        return self.transitions

    def add_markup(self, markup: Markup) -> None:
        """
        Дополнение цепей на основе разметки.

        :param markup: разметка.
        """
        words = []
        for line in markup.lines:
            for word in line.words:
                is_added = self.vocabulary.add_word(word)
                if is_added:
                    self.transitions.append(Counter())
                words.append(self.vocabulary.get_word_index(word))

        # Генерируем переходы.
        self.generate_chain(list(reversed(words)))

    def get_model(self, word_indices: List[int]) -> np.array:
        """
        Получение языковой модели.

        :param word_indices: индексы предыдущих слов.
        :return: языковая модель (распределение вероятностей для следующего слова).
        """
        l = len(self.transitions)
        if len(word_indices) == 0 or len(self.transitions[word_indices[-1]]) == 0:
            model = np.full(len(self.transitions), 1/l, dtype=np.float)
        else:
            transition = self.transitions[word_indices[-1]]
            s = sum(transition.values())
            model = np.zeros(len(self.transitions), dtype=np.float)
            for index, p in transition.items():
                model[index] = p/s
        return model
