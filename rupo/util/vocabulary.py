# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Индексы слов для языковой модели.

from typing import Dict, List, Set

from rupo.main.markup import Word


class Vocabulary(object):
    """
    Индексированный словарь.
    """
    def __init__(self) -> None:
        self.word_to_index = {}  # type: Dict[str, int]
        self.words = []  # type: List[Word]
        self.shorts_set = set()  # type: Set[str]

    def add_word(self, word: Word) -> bool:
        """
        Добавление слова.

        :param word: слово.
        :return: слово новое или нет.
        """
        short = word.get_short()
        if short not in self.shorts_set:
            self.words.append(word)
            self.shorts_set.add(short)
            self.word_to_index[short] = len(self.words)-1
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
        return self.words[index]

    def shrink(self, short_words: List[str]) -> None:
        """
        Обрезать словарь по заданным коротким формам слов.

        :param short_words: короткие формы слов.
        """
        new_words = []
        new_shorts_set = set()
        short_words = set(short_words)
        for word in self.words:
            short = word.get_short()
            if short in short_words:
                new_words.append(word)
                new_shorts_set.add(short)
                self.word_to_index[short] = len(new_words)-1
        self.shorts_set = new_shorts_set
        self.words = new_words

