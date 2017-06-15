# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль марковских цепей.

import os
import pickle
import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

from rupo.files.reader import Reader, FileType
from rupo.main.markup import Markup
from rupo.main.vocabulary import Vocabulary
from rupo.generate.model_container import ModelContainer


class MarkovModelContainer(ModelContainer):
    """
    Марковские цепи.
    """
    def __init__(self, dump_filename: str, vocabulary: Vocabulary, markup_dump_path: str=None,
                 n_poems: int=None, n_grams: int=2):
        self.n_grams = n_grams
        self.transitions = defaultdict(Counter)  # type: Dict[Tuple, Counter]
        self.vocabulary = vocabulary
        self.dump_filename = dump_filename

        # Делаем дамп модели для ускорения загрузки.
        if os.path.isfile(dump_filename):
            self.load()
        else:
            i = 0
            markups = Reader.read_markups(markup_dump_path, FileType.XML, is_processed=True)
            for markup in markups:
                self.add_markup(markup)
                i += 1
                if n_poems is not None and n_poems == i:
                    break
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

    def generate_chain(self, words: List[int]) -> Dict[Tuple, Counter]:
        """
        Генерация переходов в марковских цепях с учётом частотности.

        :param words: вершины цепи.
        :return: обновленные переходы.
        """
        for i in range(len(words) - self.n_grams + 1):
            current_words = tuple([words[j] for j in range(i, i+self.n_grams-1)])
            next_word = words[i+1]
            self.transitions[current_words][next_word] += 1
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
        l = self.vocabulary.size()
        if len(word_indices) < self.n_grams - 1:
            return np.full(self.vocabulary.size(), 1/l, dtype=np.float)
        prev_words = tuple([word_indices[-i] for i in range(1, self.n_grams)])
        if len(self.transitions[prev_words]) == 0:
            return np.full(self.vocabulary.size(), 1/l, dtype=np.float)
        else:
            transition = self.transitions[prev_words]
            s = sum(transition.values())
            model = np.zeros(self.vocabulary.size(), dtype=np.float)
            for index, p in transition.items():
                model[index] = p/s
            return model
