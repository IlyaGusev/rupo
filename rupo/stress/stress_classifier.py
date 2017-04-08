# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Расстановка ударений на основе машинного обучения.

import os
from typing import List, Tuple

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from rupo.stress.dict import StressDict
from rupo.main.markup import Syllable
from rupo.main.phonetics import Phonetics
from rupo.util.preprocess import CYRRILIC_LOWER_CONSONANTS, CYRRILIC_LOWER_VOWELS, VOWELS
from rupo.util.preprocess import get_first_vowel_position
from rupo.settings import CLASSIFIER_PATH


class MLStressClassifier:
    """
    Классификатор ударений на основе машинного обучения.
    """

    clf_filename = "clf_{}.pickle"

    def __init__(self, stress_dict: StressDict) -> None:
        """
        :param stress_dict: словарь ударений.
        """
        self.max_syllables = 12
        if not os.path.exists(CLASSIFIER_PATH):
            os.mkdir(CLASSIFIER_PATH)
        for l in range(2, self.max_syllables + 1):
            if not os.path.isfile(os.path.join(CLASSIFIER_PATH, self.clf_filename.format(l))):
                self.__build_stress_classifier(CLASSIFIER_PATH, stress_dict, l)
        self.classifiers = {l: joblib.load(os.path.join(CLASSIFIER_PATH,
                                                        self.clf_filename.format(l)))
                            for l in range(2, self.max_syllables + 1)}

    def do_cross_val(self, stress_dict: StressDict) -> List[float]:
        """
        Кроссвалидация классификаторов.

        :param stress_dict: словарь ударений.
        :return result: среднее по кроссвалидации каждого классификатора.
        """
        result = []
        for l in range(2, self.max_syllables + 1):
            train_data, answers = MLStressClassifier.__prepare_data(stress_dict, l)
            cv = ShuffleSplit(2, test_size=0.2, random_state=10)
            cv_scores = cross_val_score(self.classifiers[l], train_data, answers, cv=cv, scoring='accuracy')
            result.append(cv_scores.mean())
            print("Cross-validation with " + str(l) + " syllables: " + str(cv_scores.mean()))
        return result

    def classify_stress(self, word: str) -> int:
        """
        Проставление ударения в слове на основе классификатора.

        :param word: слово, в котором надо поставить ударение.
        :return: позиция буквы, на которую падает ударение.
        """
        syllables = Phonetics.get_word_syllables(word)
        answer = self.classifiers[len(syllables)].predict(np.array(self.__generate_sample(syllables)).reshape(1, -1))
        syllable = syllables[answer[0]]
        return get_first_vowel_position(syllable.text) + syllable.begin

    @staticmethod
    def __build_stress_classifier(model_dir: str, stress_dict: StressDict, syllables_count: int) -> None:
        """
        Построение классификатора для заданного количества слогов.

        :param model_dir: папка, куда сохранить дамп.
        :param stress_dict: словарь ударений.
        :param syllables_count: количество слогов в обучающих примерах.
        """
        train_data, answers = MLStressClassifier.__prepare_data(stress_dict, syllables_count)
        clf = DecisionTreeClassifier()
        clf.fit(train_data, answers)
        joblib.dump(clf, os.path.join(model_dir, MLStressClassifier.clf_filename.format(syllables_count)))
        print("Built stress classifier for {syllables_count} syllables".format(syllables_count=syllables_count))

    @staticmethod
    def __prepare_data(stress_dict: StressDict, syllables_count: int) -> Tuple[List[List[int]], List[int]]:
        """
        Подготовка данных для машинного обучения на оснвое словаря.

        :param stress_dict: словарь ударений.
        :param syllables_count: количество слогов, которое определяет классификатор.
        :return: X и y.
        """
        answers = []
        train_data = []
        for key, stresses in stress_dict.get_all():
            syllables = Phonetics.get_word_syllables(key)
            if len(syllables) != syllables_count:
                continue
            for syllable in syllables:
                if "ё" in key:
                    continue
                primary_stresses = [i[0] for i in stresses if i[1] == StressDict.StressType.PRIMARY]
                if len(primary_stresses) != 1:
                    continue
                for stress in primary_stresses:
                    if syllable.begin <= stress < syllable.end:
                        syllable.accent = stress
                        answers.append(syllable.number)
                        train_data.append(MLStressClassifier.__generate_sample(syllables))
        return train_data, answers

    @staticmethod
    def __generate_sample(syllables: List[Syllable]) -> List[int]:
        """
        По данному списку слогов формируется пример для машинного обучения.

        :param syllables: список слогов.
        :return: признаки.
        """
        l = len(syllables)
        features = []
        for i in range(0, l):
            text = syllables[i].text
            for j, ch in enumerate(text):
                if ch in VOWELS:
                    features.append(j)
            features.append(len(text))
            for ch1 in CYRRILIC_LOWER_CONSONANTS + CYRRILIC_LOWER_VOWELS:
                features.append(sum([ch1 == ch2 for ch2 in text]))
        return features
