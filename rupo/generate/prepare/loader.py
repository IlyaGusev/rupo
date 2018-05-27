# -*- coding: utf-8 -*-
# Автор: Даниил Анастасьев
# Описание: Загрузка словарей из корпуса.

import os
import sys
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Set

from rupo.generate.prepare.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.prepare.word_form import WordForm, LemmaCase
from rupo.generate.prepare.word_form_vocabulary import WordFormVocabulary, SEQ_END, SEQ_END_WF
from rupo.util.tqdm_open import tqdm_open


class CorporaInformationLoader(object):
    """
    Класс для построения GrammemeVectorizer и WordFormVocabulary по корпусу
    """
    def __init__(self, gram_dump_path, word_form_vocab_dump_path):
        self.grammeme_vectorizer = GrammemeVectorizer()
        if os.path.isfile(gram_dump_path):
            self.grammeme_vectorizer.load(gram_dump_path)
        self.word_form_vocabulary = WordFormVocabulary()
        if os.path.isfile(word_form_vocab_dump_path):
            self.word_form_vocabulary.load(word_form_vocab_dump_path)
        self.lemma_to_word_forms = defaultdict(set)  # type: Dict[str, Set[WordForm]]
        self.lemma_case = {}
        self.lemma_counter = Counter()  # type: Counter

    def parse_corpora(self, filenames: List[str]) -> Tuple[WordFormVocabulary, GrammemeVectorizer]:
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

        self.__add_seq_end()
        self.grammeme_vectorizer.init_possible_vectors()
        self.word_form_vocabulary.init_by_vocabulary(self.lemma_counter, self.lemma_to_word_forms, self.lemma_case)
        self.word_form_vocabulary.lemma_indices[SEQ_END_WF] = 1
        return self.word_form_vocabulary, self.grammeme_vectorizer

    def __add_seq_end(self):
        self.lemma_to_word_forms[SEQ_END].add(SEQ_END_WF)
        self.lemma_case[SEQ_END] = SEQ_END_WF.case
        self.lemma_counter[SEQ_END] = sys.maxsize

    def __process_line(self, line: str) -> None:
        try:
            text, lemma, pos_tag, grammemes = line.strip().split("\t")[:4]
            lemma = lemma.lower() + '_' + pos_tag
            gram_vector_index = self.grammeme_vectorizer.add_grammemes(pos_tag, grammemes)
            self.lemma_to_word_forms[lemma].add(WordForm(lemma, gram_vector_index, text.lower()))
            self.lemma_counter[lemma] += 1
            self.__update_lemma_case(lemma, text)
        except ValueError:
            pass

    def __update_lemma_case(self, lemma: str, text: str) -> None:
        if lemma not in self.lemma_case:
            self.lemma_case[lemma] = LemmaCase.UPPER_CASE if text.isupper() else \
                              LemmaCase.PROPER_CASE if text[0].isupper() else LemmaCase.NORMAL_CASE
        elif self.lemma_case[lemma] == LemmaCase.UPPER_CASE:
            if not text.isupper():
                self.lemma_case[lemma] = LemmaCase.PROPER_CASE if text[0].isupper() else LemmaCase.NORMAL_CASE
        elif self.lemma_case[lemma] == LemmaCase.PROPER_CASE:
            if not text[0].isupper():
                self.lemma_case[lemma] = LemmaCase.NORMAL_CASE
