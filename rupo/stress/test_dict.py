# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для словаря ударений.

import unittest

from rupo.stress.dict import StressDict
from rupo.stress.word import Stress, StressedWord
from rupo.util.preprocess import VOWELS
from rupo.settings import RU_GRAPHEME_STRESS_PATH, ZALYZNYAK_DICT, RU_GRAPHEME_STRESS_TRIE_PATH


class TestStressDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dict = StressDict(language="ru", zalyzniak_dict=ZALYZNYAK_DICT,
                              raw_dict_path=RU_GRAPHEME_STRESS_PATH, trie_path=RU_GRAPHEME_STRESS_TRIE_PATH)

    @classmethod
    def tearDownClass(cls):
        del cls.dict

    def test_get_stresses(self):
        self.assertCountEqual(self.dict.get_stresses("данный", Stress.Type.PRIMARY), [1])
        self.assertCountEqual(self.dict.get_stresses("союза", Stress.Type.PRIMARY), [2])
        self.assertCountEqual(self.dict.get_stresses("англосакс", Stress.Type.SECONDARY), [0])
        self.assertCountEqual(self.dict.get_stresses("англосакс", Stress.Type.ANY), [0, 6])
        self.assertCountEqual(self.dict.get_stresses("пора", Stress.Type.PRIMARY), [1, 3])

    def test_stress_only_in_vowels(self):
        for word, stresses in self.dict.get_all():
            for stress in stresses:
                self.assertIn(word[stress.position], VOWELS)

