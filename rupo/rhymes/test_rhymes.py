# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для модуля рифм.

import unittest

from rupo.stress.word import StressedWord, Stress
from rupo.rhymes.rhymes import Rhymes


class TestRhymes(unittest.TestCase):
    def test_rhyme(self):
        self.assertTrue(Rhymes.is_rhyme(StressedWord("тишь", {Stress(1)}),
                                        StressedWord("грустишь", {Stress(5)})))
        self.assertFalse(Rhymes.is_rhyme(StressedWord("наизусть", {Stress(4)}),
                                         StressedWord("сестра", {Stress(5)})))