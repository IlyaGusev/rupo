# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для словаря ударений.

import unittest
import os

from rupo.stress.dict import StressDict
from rupo.util.preprocess import VOWELS
from rupo.settings import RU_GRAPHEME_STRESS_PATH, ZALYZNYAK_DICT, RU_GRAPHEME_STRESS_TRIE_PATH


class TestStressDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dict = StressDict(zalyzniak_dict="/home/yallen/Документы/Python/rupo/temp_data/dict/zaliznyak.txt",
                              raw_dict_path="/home/yallen/Документы/Python/rupo/temp_data/dict/ru_grapheme_stress.txt",
                              trie_path="/home/yallen/Документы/Python/rupo/temp_data/dict/ru_grapheme_trie.pickle")

    # def test_load_and_create(self):
    #     self.assertTrue(os.path.exists(ZALYZNYAK_DICT))
    #     self.assertTrue(os.path.exists(RU_GRAPHEME_STRESS_PATH))
    #     os.remove(RU_GRAPHEME_STRESS_TRIE_PATH)
    #     StressDict()
    #     self.assertTrue(os.path.exists(RU_GRAPHEME_STRESS_TRIE_PATH))

    def test_get_stresses(self):
        self.assertCountEqual(self.dict.get_stresses("данный", StressDict.StressType.PRIMARY), [1])
        self.assertCountEqual(self.dict.get_stresses("союза", StressDict.StressType.PRIMARY), [2])
        self.assertCountEqual(self.dict.get_stresses("англосакс", StressDict.StressType.SECONDARY), [0])
        self.assertCountEqual(self.dict.get_stresses("англосакс", StressDict.StressType.ANY), [0, 6])
        self.assertCountEqual(self.dict.get_stresses("пора", StressDict.StressType.PRIMARY), [1, 3])

    def test_stress_only_in_vowels(self):
        for word, stresses in self.dict.get_all():
            for stress in stresses:
                self.assertIn(word[stress[0]], VOWELS)

