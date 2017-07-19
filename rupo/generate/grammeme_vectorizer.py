# -*- coding: utf-8 -*-
# Авторы: Гусев Илья, Анастасьев Даниил
# Описание: Модуль векторизатора граммем.

import pickle
import os
from collections import defaultdict
from typing import Dict, List, Set

from rupo.settings import GENERATOR_GRAM_VECTORS
from rupo.generate.tqdm_open import tqdm_open


def get_empty_category():
    return {GrammemeVectorizer.UNKNOWN_VALUE}


class GrammemeVectorizer:
    """
    Класс, который собирает возможные грамматические значения по корпусу и на их основе строит грамматические вектора.
    """
    UNKNOWN_VALUE = "Unknown"

    def __init__(self, dump_filename: str=GENERATOR_GRAM_VECTORS):
        """
        :param dump_filename: путь к дампу.
        """
        self.all_grammemes = defaultdict(get_empty_category)  # type: Dict[str, Set]
        self.vectors = []  # type: List[List[int]]
        self.name_to_index = {}  # type: Dict[str, int]
        self.dump_filename = dump_filename  # type: str
        if os.path.exists(self.dump_filename):
            self.load()

    def save(self) -> None:
        with open(self.dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.dump_filename, "rb") as f:
            vectorizer = pickle.load(f)
            self.__dict__.update(vectorizer.__dict__)

    def collect_grammemes(self, filename: str) -> None:
        """
        Собрать возможные грамматические значения по файлу с морфоразметкой.
        
        :param filename: файл с морфоразметкой.
        """
        with tqdm_open(filename, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                pos_tag, grammemes = line.split("\t")[2:4]
                self.add_grammemes(pos_tag, grammemes)

    def add_grammemes(self, pos_tag: str, grammemes: str) -> int:
        """
        Добавить новое грамматическое значение в список известных
        """
        vector_name = pos_tag + '#' + grammemes
        if vector_name not in self.name_to_index:
            self.name_to_index[vector_name] = len(self.name_to_index)
            self.all_grammemes["POS"].add(pos_tag)
            grammemes = grammemes.split("|") if grammemes != "_" else []
            for grammeme in grammemes:
                category = grammeme.split("=")[0]
                value = grammeme.split("=")[1]
                self.all_grammemes[category].add(value)
            self.is_initialized = False
        return self.name_to_index[vector_name]

    def init_possible_vectors(self) -> None:
        """
        Инициализировать все возможные векторы по известным грамматическим значениям
        """
        self.vectors = []
        for grammar_val, index in sorted(self.name_to_index.items(), key=lambda x: x[1]):
            pos_tag, grammemes = grammar_val.split('#')
            grammemes = grammemes.split("|") if grammemes != "_" else []
            vector = self.__build_vector(pos_tag, grammemes)
            self.vectors.append(vector)
        self.is_initialized = True

    def get_vector(self, vector_name: str) -> List[int]:
        """
        Получить вектор по грамматическим значениям.
        
        :param vector_name: часть речи + грамматическое значение.
        :return: вектор.
        """
        if vector_name not in self.name_to_index:
            return [0] * len(self.vectors[0])
        return self.vectors[self.name_to_index[vector_name]]

    def get_vector_by_index(self, index: int) -> List[int]:
        """
        Получить вектор по индексу
        
        :param vector_index
        :return: вектор.
        """
        return self.vectors[index] if 0 <= index < len(self.vectors) else [0] * len(self.vectors[0])

    def get_ordered_grammemes(self) -> List[str]:
        """
        Получить упорядоченный список возможных грамматических значений.
        
        :return: список грамматических значений.
        """
        flat = []
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            for value in sorted(list(values)):
                flat.append(category+"="+value)
        return flat

    def grammemes_count(self) -> int:
        return len(self.get_ordered_grammemes())

    def is_empty(self) -> int:
        return len(self.vectors) == 0

    def get_name_by_index(self, index):
        d = {index: name for name, index in self.name_to_index.items()}
        return d[index]

    def __build_vector(self, pos_tag: str, grammemes: List[str]) -> List[int]:
        """
        Построение вектора по части речи и грамматическим значениям.
        
        :param pos_tag: часть речи.
        :param grammemes: грамматические значения.
        :return: вектор.
        """
        vector = []
        gram_tags = {pair.split("=")[0]: pair.split("=")[1] for pair in grammemes}
        gram_tags["POS"] = pos_tag
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            if category not in gram_tags:
                vector += [1 if value == GrammemeVectorizer.UNKNOWN_VALUE else 0 for value in sorted(list(values))]
            else:
                vector += [1 if value == gram_tags[category] else 0 for value in sorted(list(values))]
        return vector
