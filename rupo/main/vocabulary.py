# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Индексы слов для языковой модели.

from typing import Dict
import pickle

from allennlp.data.vocabulary import Vocabulary

from rupo.main.markup import Markup
from rupo.files.reader import Reader, FileType
from rupo.stress.word import StressedWord, Stress
from rupo.stress.predictor import StressPredictor


class StressVocabulary(object):
    """
    Индексированный словарь.
    """
    def __init__(self) -> None:
        self.word_to_index = {}  # type: Dict[StressedWord, int]
        self.index_to_word = {}  # type: Dict[int, StressedWord]

    def save(self, dump_filename: str) -> None:
        """
        Сохранение словаря.
        """
        with open(dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dump_filename: str):
        """
        Загрузка словаря.
        """
        with open(dump_filename, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)

    def parse(self, markup_path: str, from_voc: bool=False):
         if from_voc:
            word_indexes = Reader.read_vocabulary(markup_path)
            for word, index in word_indexes:
                self.add_word(word.to_stressed_word(), index)
         else:
            markups = Reader.read_markups(markup_path, FileType.XML, is_processed=True)
            for markup in markups:
                self.add_markup(markup)

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
            if index != -1:
                self.index_to_word[index] = word
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


def inflate_stress_vocabulary(vocabulary: Vocabulary, stress_predictor: StressPredictor):
    vocab = StressVocabulary()
    for index, word in vocabulary.get_index_to_token_vocabulary("tokens").items():
        stresses = [Stress(pos, Stress.Type.PRIMARY) for pos in stress_predictor.predict(word)]
        word = StressedWord(word, set(stresses))
        vocab.add_word(word, index)
    return vocab
