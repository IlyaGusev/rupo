# -*- coding: utf-8 -*-
# Автор: Даниил Анастасьев
# Описание: Загрузка словарей из корпуса.

from collections import Counter
from typing import List, Tuple
import pickle
import os

import pymorphy2
from russian_tagsets import converters

from rupo.generate.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.tqdm_open import tqdm_open


class WordVocabulary:
    def __init__(self, dump_filename):
        self.dump_filename = dump_filename
        self.words = []
        self.word_to_index = {}
        self.counter = Counter()
        if os.path.exists(self.dump_filename):
            self.load()

    def add_word(self, word):
        if word in self.word_to_index:
            self.counter[word] += 1
        else:
            self.words.append(word)
            self.counter[word] = 1
            self.word_to_index[word] = len(self.words) - 1

    def sort(self):
        self.words = []
        self.word_to_index = {}
        for word, _ in self.counter.most_common():
            self.words.append(word)
            self.word_to_index[word] = len(self.words) - 1

    def is_empty(self):
        return len(self.words) == 0

    def save(self) -> None:
        """
        Сохранение словаря.
        """
        with open(self.dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self) -> None:
        """
        Загрузка словаря.
        """
        with open(self.dump_filename, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)


def process_tag(to_ud, tag, text):
    ud_tag = to_ud(str(tag), text)
    pos = ud_tag.split()[0]
    gram = ud_tag.split()[1].split("|")
    dropped = ["Animacy", "Aspect", "NumType"]
    gram = [grammem for grammem in gram if sum([drop in grammem for drop in dropped]) == 0]
    return pos, "|".join(sorted(gram))


class Loader(object):
    """
    Класс для построения GrammemeVectorizer и WordFormVocabulary по корпусу
    """
    def __init__(self, gram_dump_path, word_dump_path):
        self.grammeme_vectorizer = GrammemeVectorizer(gram_dump_path)
        self.word_vocabulary = WordVocabulary(word_dump_path)
        self.morph = pymorphy2.MorphAnalyzer()

    def parse_corpora(self, filenames: List[str]) -> Tuple[GrammemeVectorizer, WordVocabulary]:
        """
        Построить WordFormVocabulary, GrammemeVectorizer по корпусу

        :param filenames: пути к файлам корпуса.
        """
        for filename in filenames:
            with tqdm_open(filename, encoding="utf-8") as f:
                for line in f:
                    if line == "\n":
                        continue
                    self.__process_line(line)

        self.grammeme_vectorizer.init_possible_vectors()
        return self.grammeme_vectorizer, self.word_vocabulary

    def __process_line(self, line: str) -> None:
        text, lemma, pos_tag, grammemes = line.strip().split("\t")[:4]
        self.word_vocabulary.add_word(text.lower())
        self.grammeme_vectorizer.add_grammemes(pos_tag, grammemes)
        to_ud = converters.converter('opencorpora-int', 'ud14')
        for parse in self.morph.parse(text):
            pos, gram = process_tag(to_ud, parse.tag, text)
            self.grammeme_vectorizer.add_grammemes(pos, gram)