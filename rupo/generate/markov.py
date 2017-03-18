# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль марковских цепей.

import os
import pickle
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
    def __init__(self, dump_filename: str, vocabulary: Vocabulary, markup_dump_path: str=None):
        self.transitions = list()
        self.vocabulary = vocabulary
        self.dump_filename = dump_filename

        # Делаем дамп модели для ускорения загрузки.
        if os.path.isfile(dump_filename):
            self.load()
        else:
            for i in range(len(self.vocabulary.words)):
                self.transitions.append(Counter())
            i = 0
            markups = Reader.read_markups(markup_dump_path, FileTypeEnum.XML, is_processed=True)
            for markup in markups:
                self.add_markup(markup)
                i += 1
                if i % 500 == 0:
                    print(i)
            self.save()

    def save(self):
        with open(self.dump_filename, "wb") as f:
            pickle.dump(self.transitions, f, pickle.HIGHEST_PROTOCOL)
        self.vocabulary.save()

    def load(self):
        with open(self.dump_filename, "rb") as f:
            self.transitions = pickle.load(f)
        self.vocabulary.load()

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
                try:
                    self.vocabulary.get_word_index(word)
                    words.append(self.vocabulary.get_word_index(word))
                except IndexError:
                    print("Слово не из словаря.")
                    pass

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
