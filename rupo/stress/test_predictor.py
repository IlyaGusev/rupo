# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты предсказателя ударений.

import unittest

from rupo.stress.predictor import CombinedStressPredictor


class TestStressPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress_predictor = CombinedStressPredictor()

    def test_stress(self):
        checks = {
            'я': [0],
            'в': [],
            'он': [0],
            'майка': [1],
            'соломка': [3],
            'изжить': [3],
            'виться': [1],
            'данный': [1],
            'зорька': [1],
            'банка': [1],
            'оттечь': [3],
            'советского': [3],
            'союза': [2],
            'пора': [3, 1],
            'изжила': [5],
            'меда': [1],
            'автоподъёмник': [8],
        }
        for word, pos in checks.items():
            print(word)
            self.assertEqual(sorted(self.stress_predictor.predict(word)), sorted(pos))


