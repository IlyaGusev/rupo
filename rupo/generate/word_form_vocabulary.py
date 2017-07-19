# -*- coding: utf-8 -*-
# Авторы: Гусев Илья, Анастасьев Даниил
# Описание: Словарь словоформ.

import pickle
import os
from collections import defaultdict, Counter
from typing import List, Dict
from tqdm import tqdm

from rupo.generate.word_form import WordForm, LemmaCase
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
        self.word_forms = []  # type: List[WordForm]`
        self.word_form_indices = {}  # WordForm -> index в self.word_forms
        self.word_form_indices_rev = {}  # index в self.word_forms -> WordForm
        self.lemma_indices = {}  # Lemma -> index
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

    def init_by_vocabulary(self, lemma_counter: Counter, lemma_to_word_forms: Dict[str, List[WordForm]], 
                           lemma_case: Dict[str, LemmaCase]):
        """
        Строит словарь по предподсчитанным данным
        
        :param lemma_counter: Counter по леммам.
        :param lemma_to_word_forms: Отображение из леммы в список известных словоформ для неё (её парадигму)
        :param lemma_case: Отображение из леммы в тип капитализации, известный для этой леммы
        """
        for i, (lemma, _) in enumerate(tqdm(lemma_counter.most_common(), desc="Init vocabulary")):
            for word_form in lemma_to_word_forms[lemma]:
                word_form.set_case(lemma_case[word_form.lemma])
                self.word_forms.append(word_form)
                self.word_form_indices[word_form] = len(self.word_forms)
                self.lemma_indices[word_form] = i + 1
        self.word_form_indices_rev = {self.word_form_indices[x] : x for x in self.word_form_indices}

    def get_word_form_index(self, word_form: WordForm) -> int:
        return self.word_form_indices[word_form]
            
    def get_word_form_by_index(self, index: int) -> WordForm:
        return self.word_forms[index]

    def get_word_form_index_min(self, word_form: WordForm, size: int) -> int:
        return min(self.get_word_form_index(word_form), size)

    def get_lemma_index(self, word_form: WordForm) -> int:
        return self.lemma_indices[word_form]

    def get_sequence_end_index(self, seq_end: WordForm) -> int:
        """
        Возвращает индекс завершающего строку символа. Предполагается, что он последний в lemma_indices
        """
        assert seq_end in self.lemma_indices and self.lemma_indices[seq_end] == len(self.lemma_indices) - 1
        return len(self.lemma_indices) - 1

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
