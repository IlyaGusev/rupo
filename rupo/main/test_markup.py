# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для разметки.

import unittest

from rupo.util.data import MARKUP_EXAMPLE
from rupo.main.markup import Markup
from rupo.stress.predictor import CombinedStressPredictor
from rupo.settings import RU_STRESS_DEFAULT_MODEL, ZALYZNYAK_DICT, CMU_DICT, \
    RU_GRAPHEME_STRESS_PATH, RU_GRAPHEME_STRESS_TRIE_PATH


class TestMarkup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress_predictor = CombinedStressPredictor(
            stress_model_path=RU_STRESS_DEFAULT_MODEL,
            zalyzniak_dict=ZALYZNYAK_DICT,
            cmu_dict=CMU_DICT,
            raw_stress_dict_path=RU_GRAPHEME_STRESS_PATH,
            stress_trie_path=RU_GRAPHEME_STRESS_TRIE_PATH
        )
        # cls.stress_predictor = DictStressPredictor(raw_dict_path=RU_GRAPHEME_STRESS_PATH,
        #                                            trie_path=RU_GRAPHEME_STRESS_TRIE_PATH,
        #                                            zalyzniak_dict=ZALYZNYAK_DICT, cmu_dict=CMU_DICT)

    @classmethod
    def tearDownClass(cls):
        del cls.stress_predictor

    def test_from_to(self):
        clean_markup = Markup()
        self.assertEqual(MARKUP_EXAMPLE, clean_markup.from_xml(MARKUP_EXAMPLE.to_xml()))
        clean_markup = Markup()
        self.assertEqual(MARKUP_EXAMPLE, clean_markup.from_json(MARKUP_EXAMPLE.to_json()))

    def test_process_text(self):
        text = "Соломка король себя.\n Пора виться майкой в."
        markup = Markup.process_text(text, self.stress_predictor)
        print(markup)
        self.assertEqual(markup, MARKUP_EXAMPLE)

