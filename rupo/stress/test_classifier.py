# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для модуля классификации ударений.

import unittest

from rupo.stress.stress_classifier import MLStressClassifier
from rupo.stress.dict import StressDict


class TestStressClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress_dict = StressDict()
        cls.stress_classifier = MLStressClassifier(cls.stress_dict)

    def test_classifier(self):
        self.assertEqual(len([0, 2, 2, 1, 2, 2, 2, 1, 2]),
                         len([self.stress_classifier.classify_stress(word) for word in
                              ["волки", "перелив", "карачун", "пипярка", "пепелац", "гиппогриф",
                              "хвосторог", "стартап", "квинтолап"]]))
