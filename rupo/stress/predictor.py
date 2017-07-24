# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс для определения ударения.

import os
from typing import List
from rupo.stress.rnn import RNNStressModel
from rupo.stress.dict import StressDict
from rupo.g2p.rnn import RNNG2PModel
from rupo.settings import RU_STRESS_DEFAULT_MODEL, EN_STRESS_DEFAULT_MODEL, RU_G2P_DEFAULT_MODEL, EN_G2P_DEFAULT_MODEL
from rupo.g2p.aligner import Aligner
from rupo.util.preprocess import count_vowels, get_first_vowel_position
from rupo.settings import CMU_DICT, ZALYZNYAK_DICT, RU_GRAPHEME_SET, RU_WIKI_DICT


class StressPredictor:
    def predict(self, word: str) -> List[int]:
        raise NotImplementedError()


class RNNStressPredictor(StressPredictor):
    def __init__(self, language: str="ru", stress_model_path: str=None, g2p_model_path: str=None,
                 grapheme_set=RU_GRAPHEME_SET, g2p_dict_path=None, aligner_dump_path=None,
                 ru_wiki_dict=RU_WIKI_DICT, cmu_dict=CMU_DICT):
        self.language = language
        self.stress_model_path = stress_model_path
        self.g2p_model_path = g2p_model_path

        if language == "ru":
            self.__init_language_defaults(RU_STRESS_DEFAULT_MODEL, RU_G2P_DEFAULT_MODEL)
        elif language == "en":
            self.__init_language_defaults(EN_STRESS_DEFAULT_MODEL, EN_G2P_DEFAULT_MODEL)
        else:
            raise RuntimeError("Wrong language")

        if not os.path.exists(self.stress_model_path) or not os.path.exists(self.g2p_model_path):
            raise RuntimeError("No stress or g2p models available (or wrong paths)")

        self.stress_model = RNNStressModel(language=language)
        self.stress_model.load(self.stress_model_path)
        self.g2p_model = RNNG2PModel(language=language)
        self.g2p_model.load(self.g2p_model_path)
        self.aligner = Aligner(language, grapheme_set, g2p_dict_path, aligner_dump_path,
                               ru_wiki_dict=ru_wiki_dict, cmu_dict=cmu_dict)

    def __init_language_defaults(self, stress_model_path, g2p_model_path):
        if self.stress_model_path is None:
            self.stress_model_path = stress_model_path
        if self.g2p_model_path is None:
            self.g2p_model_path = g2p_model_path

    def predict(self, word: str) -> List[int]:
        word = word.lower()
        if sum([int(ch not in self.aligner.grapheme_set) for ch in word]) != 0:
            return []
        phonemes = self.g2p_model.predict([word])[0].replace(" ", "")
        stresses = self.stress_model.predict([phonemes])[0]
        stresses = [i for i, stress in enumerate(stresses) if stress == 1 or stress == 2]
        g, p = self.aligner.align(word, phonemes)
        stresses = self.aligner.align_stresses(g, p, stresses, is_grapheme=False)
        for i, stress in enumerate(stresses):
            stresses[i] -= len([ch for ch in g[:stress] if ch == " "])
        stresses = [i for i in stresses if i < len(word)]
        return stresses


class DictStressPredictor(StressPredictor):
    def __init__(self, language="ru", raw_dict_path=None, trie_path=None,
                 zalyzniak_dict=ZALYZNYAK_DICT, cmu_dict=CMU_DICT):
        self.stress_dict = StressDict(language, raw_dict_path=raw_dict_path, trie_path=trie_path,
                                      zalyzniak_dict=zalyzniak_dict, cmu_dict=cmu_dict)

    def predict(self, word: str) -> List[int]:
        """
        Определение ударения в слове по словарю. Возможно несколько вариантов ударения.

        :param word: слово для простановки ударений.
        :return stresses: позиции букв, на которые падает ударение.
        """
        stresses = []
        if count_vowels(word) == 0:
            # Если гласных нет, то и ударений нет.
            pass
        elif count_vowels(word) == 1:
            # Если одна гласная, то на неё и падает ударение.
            stresses.append(get_first_vowel_position(word))
        elif word.find("ё") != -1:
            # Если есть буква "ё", то только на неё может падать ударение.
            stresses.append(word.find("ё"))
        else:
            # Проверяем словарь на наличие форм с ударениями.
            stresses = self.stress_dict.get_stresses(word)
            if 'е' not in word:
                return stresses
            # Находим все возможные варинаты преобразований 'е' в 'ё'.
            positions = [i for i in range(len(word)) if word[i] == 'е']
            beam = [word[:positions[0]]]
            for i in range(len(positions)):
                new_beam = []
                for prefix in beam:
                    n = positions[i+1] if i+1 < len(positions) else len(word)
                    new_beam.append(prefix + 'ё' + word[positions[i]+1:n])
                    new_beam.append(prefix + 'е' + word[positions[i]+1:n])
                    beam = new_beam
            # И проверяем их по словарю.
            for permutation in beam:
                if len(self.stress_dict.get_stresses(permutation)) != 0:
                    yo_pos = permutation.find("ё")
                    if yo_pos != -1:
                        stresses.append(yo_pos)
        return stresses


class CombinedStressPredictor(StressPredictor):
    def __init__(self, language="ru", stress_model_path: str=None, g2p_model_path: str=None,
                 grapheme_set=RU_GRAPHEME_SET, g2p_dict_path=None, aligner_dump_path=None, raw_stress_dict_path=None,
                 stress_trie_path=None, zalyzniak_dict=ZALYZNYAK_DICT, cmu_dict=CMU_DICT, ru_wiki_dict=RU_WIKI_DICT):
        self.rnn = RNNStressPredictor(language, stress_model_path, g2p_model_path, grapheme_set,
                                      g2p_dict_path, aligner_dump_path, ru_wiki_dict, cmu_dict)
        self.dict = DictStressPredictor(language, raw_stress_dict_path, stress_trie_path, zalyzniak_dict, cmu_dict)

    def predict(self, word: str) -> List[int]:
        stresses = self.dict.predict(word)
        if len(stresses) == 0:
            return self.rnn.predict(word)
        else:
            return stresses
