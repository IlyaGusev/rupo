# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс для определения ударения.

from typing import List
from rupo.stress.rnn import RNNStressModel
from rupo.stress.dict import StressDict
from rupo.g2p.rnn import RNNG2PModel
from rupo.settings import RU_STRESS_DEFAULT_MODEL, EN_STRESS_DEFAULT_MODEL, RU_G2P_DEFAULT_MODEL, EN_G2P_DEFAULT_MODEL
from rupo.g2p.aligner import Aligner
from rupo.util.preprocess import count_vowels, get_first_vowel_position


class StressPredictor:
    def predict(self, word: str) -> List[int]:
        raise NotImplementedError()


class RNNStressPredictor(StressPredictor):
    def __init__(self, language="ru"):
        self.stress_model = RNNStressModel(language=language)
        self.g2p_model = RNNG2PModel(language=language)
        if language == "ru":
            stress_model_path = RU_STRESS_DEFAULT_MODEL
            g2p_model_path = RU_G2P_DEFAULT_MODEL
        elif language == "en":
            stress_model_path = EN_STRESS_DEFAULT_MODEL
            g2p_model_path = EN_G2P_DEFAULT_MODEL
        else:
            raise RuntimeError("Wrong language")
        self.stress_model.load(stress_model_path)
        self.g2p_model.load(g2p_model_path)
        self.aligner = Aligner(language=language)

    def predict(self, word: str) -> List[int]:
        word = word.lower()
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
    def __init__(self, language="ru"):
        self.stress_dict = StressDict(language)

    def predict(self, word: str) -> List[int]:
        """
        Определение ударения в слове по словарю. Возможно несколько вариантов ударения.

        :param word: слово для простановки ударений.
        :param stress_dict: экземпляр обёртки для словаря ударений.
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
    def __init__(self, language="ru"):
        self.rnn = RNNStressPredictor(language)
        self.dict = DictStressPredictor(language)

    def predict(self, word: str) -> List[int]:
        stresses = self.dict.predict(word)
        if len(stresses) == 0:
            return self.rnn.predict(word)
        else:
            return stresses
