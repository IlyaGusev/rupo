# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс для определения ударения.

from typing import List

from rupo.stress.dict import StressDict
from rupo.util.preprocess import count_vowels, get_first_vowel_position
from rupo.settings import CMU_DICT, ZALYZNYAK_DICT, RU_STRESS_DEFAULT_MODEL
from rupo.stress.word import Stress

from russ.stress.model import StressModel


class StressPredictor:
    def predict(self, word: str) -> List[int]:
        raise NotImplementedError()


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
            stresses = self.stress_dict.get_stresses(word, Stress.Type.PRIMARY) +\
                       self.stress_dict.get_stresses(word, Stress.Type.SECONDARY)
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
    def __init__(self, language="ru", stress_model_path: str=RU_STRESS_DEFAULT_MODEL, raw_stress_dict_path=None,
                 stress_trie_path=None, zalyzniak_dict=ZALYZNYAK_DICT, cmu_dict=CMU_DICT):
        self.rnn = StressModel.load(stress_model_path)
        self.dict = DictStressPredictor(language, raw_stress_dict_path, stress_trie_path, zalyzniak_dict, cmu_dict)

    def predict(self, word: str) -> List[int]:
        stresses = self.dict.predict(word)
        if len(stresses) == 0:
            return self.rnn.predict(word)
        else:
            return stresses
