# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для разметки.

import unittest

from rupo.util.data import MARKUP_EXAMPLE
from rupo.main.markup import Markup
from rupo.stress.predictor import CombinedStressPredictor


class TestMarkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress_predictor = CombinedStressPredictor()

    def test_from_to(self):
        clean_markup = Markup()
        self.assertEqual(MARKUP_EXAMPLE, clean_markup.from_xml(MARKUP_EXAMPLE.to_xml()))
        clean_markup = Markup()
        self.assertEqual(MARKUP_EXAMPLE, clean_markup.from_json(MARKUP_EXAMPLE.to_json()))

    def test_process_text(self):
        text = "Соломка король себя.\n Пора виться майкой в."
        markup = Markup.process_text(text, self.stress_predictor)
        self.assertEqual(markup, MARKUP_EXAMPLE)

