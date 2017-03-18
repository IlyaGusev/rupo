# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для модуля фонетики.

import unittest

from rupo.main.markup import Syllable, Word
from rupo.rhymes.rhymes import Rhymes


class TestRhymes(unittest.TestCase):
    def test_rhyme(self):
        self.assertTrue(Rhymes.is_rhyme(Word(0, 4, "тишь", [Syllable(0, 4, 0, "тишь", 1)]),
                                        Word(0, 8, "грустишь", [Syllable(0, 3, 0, "гру"),
                                                                Syllable(3, 8, 1, "стишь", 5)])))
        self.assertFalse(Rhymes.is_rhyme(Word(0, 8, "наизусть", [Syllable(0, 2, 0, "на"), Syllable(2, 4, 1, "из"),
                                                                 Syllable(4, 8, 2, "усть", 4)]),
                                         Word(0, 6, "сестра", [Syllable(0, 3, 0, "сест"),
                                                               Syllable(3, 6, 1, "ра", 5)])))