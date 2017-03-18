# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для API библиотеки.

import unittest

from rupo.main.markup import Markup
from rupo.api import get_accent, get_word_syllables, count_syllables, get_markup, get_improved_markup, \
    is_rhyme, classify_metre


class TestApi(unittest.TestCase):
    def test_accent(self):
        self.assertEqual(get_accent("корова"), 3)

    def test_get_word_syllables(self):
        self.assertEqual(get_word_syllables("корова"), ["ко", "ро", "ва"])

    def test_count_syllables(self):
        self.assertEqual(count_syllables("корова"), 3)

    def test_is_rhyme(self):
        self.assertTrue(is_rhyme("корова", "здорова"))

    def test_get_markup(self):
        self.assertIsInstance(get_markup("корова"), Markup)

    def test_get_improved_markup(self):
        self.assertIsInstance(get_improved_markup("корова"), Markup)

    def test_classify_metre(self):
        text = "Горит восток зарёю новой.\n" \
               "Уж на равнине, по холмам\n" \
               "Грохочут пушки. Дым багровый\n" \
               "Кругами всходит к небесам."
        self.assertEqual(classify_metre(text), "iambos")
