# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для словаря ударений.

import unittest
import os

from rupo.accents.dict import AccentDict
from rupo.util.preprocess import VOWELS
from rupo.settings import DICT_TXT_PATH, DICT_TRIE_PATH


class TestAccentDict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dict = AccentDict()

    def test_load_and_create(self):
        self.assertTrue(os.path.exists(DICT_TXT_PATH))
        self.assertTrue(os.path.exists(DICT_TRIE_PATH))
        os.remove(DICT_TRIE_PATH)
        AccentDict()
        self.assertTrue(os.path.exists(DICT_TRIE_PATH))

    def test_get_accents(self):
        self.assertCountEqual(self.dict.get_accents("данный", AccentDict.AccentType.PRIMARY), [1])
        self.assertCountEqual(self.dict.get_accents("союза", AccentDict.AccentType.PRIMARY), [2])
        self.assertCountEqual(self.dict.get_accents("англосакс", AccentDict.AccentType.SECONDARY), [0])
        self.assertCountEqual(self.dict.get_accents("англосакс", AccentDict.AccentType.ANY), [0, 6])
        self.assertCountEqual(self.dict.get_accents("пора", AccentDict.AccentType.PRIMARY), [1, 3])

    def test_accent_only_in_vowels(self):
        for word, accents in self.dict.get_all():
            for accent in accents:
                self.assertIn(word[accent[0]], VOWELS)

