# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты фонетического разборщика.

import unittest
import logging
import sys

from rupo.g2p.predictor import RNNG2PPredictor
from rupo.settings import RU_G2P_DEFAULT_MODEL


class TestG2Predictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g2p_predictor = RNNG2PPredictor(
            language="ru",
            g2p_model_path=RU_G2P_DEFAULT_MODEL
        )
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def test_g2p(self):
        checks = {
            'я': ['ja'],
            'в': ['f'],
            'он': ['on'],
            'корова': ['kərovə', 'kərɐvə'],
            'мышь': ['mɨʂ'],
            'чрезвычайный': ['ʨrʲɪzvɨʨæjnɨj'],
            'абажур': ['əbɐʐur', 'ɐbɐʐur'],
            'лёгкий': ['lʲɵxʲkʲɪj']
        }
        for word, pos in checks.items():
            self.assertIn(self.g2p_predictor.predict(word), pos)
