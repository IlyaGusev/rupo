# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Индексы слов для языковой модели.

from typing import Dict, List, Set
import pickle
import os
import copy

from rupo.main.markup import Word, Markup
from rupo.files.reader import Reader, FileType


class Vocabulary(object):
    """
    Индексированный словарь.
    """
    def __init__(self, dump_filename: str, markup_path: str=None, from_voc: bool=False) -> None:
        """
        :param dump_filename: файл, в который сохранется словарь.
        :param markup_path: файл/папка с разметками.
        """
        self.dump_filename = dump_filename
        self.word_to_index = {}  # type: Dict[str, int]
        self.index_to_word = {}  # type: Dict[int, Word]
        self.shorts_set = set()  # type: Set[str]

        if os.path.isfile(self.dump_filename):
            self.load()
        else:
<<<<<<< HEAD
            if from_voc:
                word_indexes = Reader.read_vocabulary(markup_path)
                for word, index in word_indexes:
                    self.add_word(word, index)
            else:
                markups = Reader.read_markups(markup_path, FileType.XML, is_processed=True)
                for markup in markups:
                    self.add_markup(markup)
=======
            i = 0
            markups = Reader.read_markups(markup_path, FileType.VOCAB, is_processed=True)
            for i, markup in enumerate(markups):
                self.add_markup(markup, i)
                if i % 50 == 0:
                    print(i)
>>>>>>> 45049a4f591075476d71761bf2319d754376e847
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
                self.add_word(word)

    def add_word(self, word: Word, index: int=-1) -> bool:
        """
        Добавление слова.

        :param word: слово.
        :param index: индекс, если задан заранее.
        :return: слово новое или нет.
        """
        short = word.get_short()
        self.words.append(word)
        self.word_to_index[short] = len(self.words) if index == -1 else index
        self.index_to_word[len(self.words) if index == -1 else index] = word
        if short not in self.shorts_set:
<<<<<<< HEAD
            if index == -1:
                i = 0
                while i in self.index_to_word:
                    i += 1
                index = i
            self.index_to_word[index] = word
            self.shorts_set.add(short)
            self.word_to_index[short] = index
=======
            self.shorts_set.add(short)
>>>>>>> 45049a4f591075476d71761bf2319d754376e847
            return True
        return False

    def get_word_index(self, word: Word) -> int:
        """
        Получить индекс слова.

        :param word: слово (Word).
        :return: индекс.
        """
        short = word.get_short()
        if short in self.word_to_index:
            return self.word_to_index[short]
        raise IndexError("Can't find word: " + word.text)

    def get_word(self, index: int) -> Word:
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

    def shrink(self, short_words: List[str]) -> None:
        """
        Обрезать словарь по заданным коротким формам слов.

        :param short_words: короткие формы слов.
        """
        old_words = copy.deepcopy(self.index_to_word)
        short_words = set(short_words)
        for word in old_words.values():
            if word.get_short() in short_words:
                self.add_word(word)
