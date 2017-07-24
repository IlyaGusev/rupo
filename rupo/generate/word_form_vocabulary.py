# -*- coding: utf-8 -*-
# Авторы: Гусев Илья, Анастасьев Даниил
# Описание: Словарь словоформ.

import pickle
import os
from collections import Counter
from typing import List, Dict, Set
from tqdm import tqdm

from rupo.generate.word_form import WordForm, LemmaCase
from rupo.settings import GENERATOR_WORD_FORM_VOCAB_PATH, GENERATOR_VOCAB_PATH
from rupo.main.markup import Word
from rupo.g2p.graphemes import Graphemes

# Индикатор конца последовательности
SEQ_END = '</s>'
SEQ_END_WF = WordForm(SEQ_END, -1, SEQ_END)


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
        # WordForm -> index в self.word_forms
        self.word_form_indices = {}  # type: Dict[WordForm, int]
        # WordForm -> lemma index
        self.lemma_indices = {}  # type: Dict[WordForm, int]
        self.text_to_word_form = None
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

    def init_by_vocabulary(self, lemma_counter: Counter, lemma_to_word_forms: Dict[str, Set[WordForm]],
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
                self.word_form_indices[word_form] = len(self.word_forms) - 1
                assert self.word_forms[self.word_form_indices[word_form]] == word_form
                self.lemma_indices[word_form] = i + 1  # 0 - зарезервирован для паддинга.
        assert self.lemma_indices[SEQ_END_WF] == 1

    def get_word_form_index(self, word_form: WordForm) -> int:
        return self.word_form_indices[word_form]
            
    def get_word_form_by_index(self, index: int) -> WordForm:
        return self.word_forms[index]

    def get_word_form_index_min(self, word_form: WordForm, size: int) -> int:
        return min(self.get_word_form_index(word_form), size)

    def get_lemma_index_min(self, word_form: WordForm, size: int) -> int:
        return min(self.get_lemma_index(word_form), size)

    def get_lemma_index(self, word_form: WordForm) -> int:
        return self.lemma_indices[word_form]

    def get_sequence_end_index(self) -> int:
        """
        Возвращает индекс словоформы завершающего строку символа.
        """
        assert SEQ_END_WF in self.word_form_indices and self.word_form_indices[SEQ_END_WF] == 0
        return 0

    def get_sequence_end_lemma_index(self) -> int:
        """
        Возвращает индекс леммы завершающего строку символа.
        """
        assert SEQ_END_WF in self.lemma_indices and self.lemma_indices[SEQ_END_WF] == 1
        return 1

    def get_softmax_size_by_lemma_size(self, lemma_size: int):
        assert lemma_size + 1 < len(self.lemma_indices)
        final_lemma = SEQ_END
        for form, index in self.lemma_indices.items():
            if index == lemma_size + 1:
                final_lemma = form.lemma
                break
        for i, form in enumerate(self.word_forms):
            if form.lemma == final_lemma:
                return i
        assert False

    def inflate_vocab(self, dump_path, top_n=None) -> None:
        """
        Получение словаря с ударениями по этому словарю.
        
        :param top_n: сколько первых записей взять?
        :param dump_path: путь, куда сохранить словарь.
        """
        from rupo.main.vocabulary import Vocabulary
        from rupo.stress.predictor import CombinedStressPredictor
        vocab = Vocabulary(dump_path)
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

    def inflate_text_mappings(self):
        self.text_to_word_form = {}
        for i, form in enumerate(self.word_forms):
            self.text_to_word_form[form.text] = i

    def get_word_form_by_text(self, text):
        if self.text_to_word_form is None:
            self.inflate_text_mappings()
        return self.word_forms[self.text_to_word_form[text]]

    def is_empty(self) -> int:
        return len(self.word_forms) == 0
