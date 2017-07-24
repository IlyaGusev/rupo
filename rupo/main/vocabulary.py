# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Индексы слов для языковой модели.

from typing import Dict
import pickle
import os

from rupo.main.markup import Markup
from rupo.files.reader import Reader, FileType
from rupo.stress.word import StressedWord


class StressVocabulary(object):
    """
    Индексированный словарь.
    """
    def __init__(self, dump_filename: str, markup_path: str=None, from_voc: bool=False) -> None:
        """
        :param dump_filename: файл, в который сохранется словарь.
        :param markup_path: файл/папка с разметками.
        """
        self.dump_filename = dump_filename
        self.word_to_index = {}  # type: Dict[StressedWord, int]
        self.index_to_word = {}  # type: Dict[int, StressedWord]

        if os.path.isfile(self.dump_filename):
            self.load()
        elif markup_path is not None:
            if from_voc:
                word_indexes = Reader.read_vocabulary(markup_path)
                for word, index in word_indexes:
                    self.add_word(word.to_stressed_word(), index)
            else:
                markups = Reader.read_markups(markup_path, FileType.XML, is_processed=True)
                for markup in markups:
                    self.add_markup(markup)
            self.save()

    def save(self) -> None:
        """
        Сохранение словаря.
        """
        with open(self.dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        """
        Загрузка словаря.
        """
        with open(self.dump_filename, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)

    def add_markup(self, markup: Markup) -> None:
        """
        Добавление слов из разметки в словарь.

        :param markup: разметка.
        """
        for line in markup.lines:
            for word in line.words:
                self.add_word(word.to_stressed_word())

    def add_word(self, word: StressedWord, index: int=-1) -> bool:
        """
        Добавление слова.

        :param word: слово.
        :param index: индекс, если задан заранее.
        :return: слово новое или нет.
        """
        if word in self.word_to_index:
            return False
        self.word_to_index[word] = self.size() if index == -1 else index
        self.index_to_word[self.size() if index == -1 else index] = word
        return True

    def get_word_index(self, word: StressedWord) -> int:
        """
        Получить индекс слова.

        :param word: слово (Word).
        :return: индекс.
        """
        if word in self.word_to_index:
            return self.word_to_index[word]
        raise IndexError("Can't find word: " + word.text)

    def get_word(self, index: int) -> StressedWord:
        """
        Получить слово по индексу.

        :param index: индекс.
        :return: слово.
        """
        return self.index_to_word[index]

    def size(self):
        """
        :return: получить размер словаря.
        """
        return len(self.index_to_word)
