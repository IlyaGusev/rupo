# -*- coding: utf-8 -*-
# Авторы: Гусев Илья, Анастасьев Даниил
# Описание: Словарь словоформ.

import pickle
import os
from collections import defaultdict, Counter
from typing import List, Dict
from tqdm import tqdm

from rupo.generate.word_form import WordForm
from rupo.generate.grammeme_vectorizer import GrammemeVectorizer
from rupo.settings import GENERATOR_WORD_FORM_VOCAB_PATH, GENERATOR_VOCAB_PATH
from rupo.main.vocabulary import Vocabulary
from rupo.stress.predictor import CombinedStressPredictor
from rupo.main.markup import Word
from rupo.g2p.graphemes import Graphemes


class WordFormVocabulary(object):
    """
    Класс словаря словоформ.
    """
    def __init__(self, dump_filename: str=GENERATOR_WORD_FORM_VOCAB_PATH):
        """
        :param dump_filename: путь к дампу словаря.
        """
        self.dump_filename = dump_filename  # type: str
        self.word_forms = []  # type: List[WordForm]
        self.word_form_indices = {}  # type: Dict[WordForm, int]
        self.lemma_to_word_form_indices = defaultdict(list)  # type: Dict[str, List[int]]
        self.text_to_word_forms = defaultdict(list)
        self.lemma_counter = Counter()  # type: Counter
        self.sorted = False  # type: bool
        if os.path.exists(self.dump_filename):
            self.load()

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

    def load_from_corpus(self, filename: str, grammeme_vectorizer: GrammemeVectorizer) -> None:
        """
        Пополнение словаря по морфоразметке.
        
        :param filename: имя файла с морфоразметкой.
        :param grammeme_vectorizer: векторизатор грамматических значений.
        """
        with open(filename, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Collecting word forms"):
                if line == "\n":
                    continue
                form, lemma, pos_tag, grammemes = line.split("\t")[:4]
                form = form.lower()
                lemma = lemma.lower()
                vector_name = pos_tag + "#" + grammemes
                self.add_word_form(form, lemma + "_" + pos_tag, grammeme_vectorizer.name_to_index[vector_name])

    def add_word_form(self, text: str, lemma: str, gram_vector_index: int) -> None:
        """
        Добавление словоформы в словарь.
        
        :param text: вокабула словоформы.
        :param lemma: лемма словоформы (=начальная форма, нормальная форма).
        :param gram_vector_index: индекс грамматического вектора.
        """
        word_form = WordForm(lemma, gram_vector_index, text)
        self.lemma_counter[lemma] += 1
        if word_form not in self.word_form_indices:
            self.word_forms.append(word_form)
            index = len(self.word_forms) - 1
            self.word_form_indices[word_form] = index
            self.lemma_to_word_form_indices[lemma].append(index)
            self.text_to_word_forms[text].append(word_form)
        self.sorted = False

    def get_word_form_index(self, word_form: WordForm) -> int:
        return self.word_form_indices[word_form]

    def get_word_form_by_index(self, index: int) -> WordForm:
        return self.word_forms[index]

    def get_word_form_index_min(self, word_form: WordForm, size: int) -> int:
        return min(self.get_word_form_index(word_form), size)

    def get_word_forms_by_text(self, text: str) -> List[WordForm]:
        return self.text_to_word_forms[text]

    def sort(self) -> None:
        """
        Сортировка по частотности лемм.
        """
        new_vocab = WordFormVocabulary()
        for lemma, _ in tqdm(self.lemma_counter.most_common(), desc="Sorting vocabulary"):
            for index in self.lemma_to_word_form_indices[lemma]:
                word_form = self.word_forms[index]
                new_vocab.add_word_form(word_form.text, word_form.lemma, word_form.gram_vector_index)
        self.word_forms = new_vocab.word_forms
        self.word_form_indices = new_vocab.word_form_indices
        self.lemma_to_word_form_indices = new_vocab.lemma_to_word_form_indices
        self.text_to_word_forms = new_vocab.text_to_word_forms
        self.sorted = True

    def inflate_vocab(self, top_n=None) -> None:
        """
        Получение словаря с ударениями по этому словарю.
        :param top_n: сколько первых записей взять?
        """
        vocab = Vocabulary(GENERATOR_VOCAB_PATH)
        stress_predictor = CombinedStressPredictor()
        forms = self.word_forms
        if top_n is not None:
            forms = forms[:top_n]
        for index, word_form in tqdm(enumerate(forms), desc="Accenting words"):
            text = word_form.text
            stresses = stress_predictor.predict(text)
            word = Word(-1, -1, text, Graphemes.get_syllables(text))
            word.set_stresses(stresses)
            vocab.add_word(word, index)
        vocab.save()

    def is_empty(self) -> int:
        return len(self.word_forms) == 0
