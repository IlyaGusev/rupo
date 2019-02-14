# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для API библиотеки.

import unittest
import os
import random

from rupo.settings import MARKUP_XML_EXAMPLE, EXAMPLES_DIR, GENERATOR_MODEL_DIR, \
    GENERATOR_WORD_FORM_VOCAB_PATH, GENERATOR_VOCAB_PATH, RU_STRESS_DEFAULT_MODEL,\
    ZALYZNYAK_DICT
from rupo.main.markup import Markup
from rupo.api import Engine


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(language="ru")
        cls.engine.load(
            stress_model_path=RU_STRESS_DEFAULT_MODEL,
            zalyzniak_dict=ZALYZNYAK_DICT
        )

    @classmethod
    def tearDownClass(cls):
        del cls.engine

    def test_stress(self):
        self.assertEqual(self.engine.get_stresses("корова"), [3])
        self.assertCountEqual(self.engine.get_stresses("авиамоделирование"), [0, 9])
        self.assertEqual(self.engine.get_stresses("триплекс"), [2])
        self.assertEqual(self.engine.get_stresses("горит"), [3])
        self.assertEqual(self.engine.get_stresses("восток"), [4])
        self.assertEqual(self.engine.get_stresses("зарёю"), [3])
        self.assertEqual(self.engine.get_stresses("новой"), [1])
        self.assertEqual(self.engine.get_stresses("равнине"), [4])
        self.assertEqual(self.engine.get_stresses("холмам"), [4])
        self.assertEqual(self.engine.get_stresses("грохочут"), [4])
        self.assertCountEqual(self.engine.get_stresses("пушки"), [4, 1])
        self.assertEqual(self.engine.get_stresses("багровый"), [4])
        self.assertEqual(self.engine.get_stresses("кругами"), [4])
        self.assertEqual(self.engine.get_stresses("уж"), [0])
        self.assertEqual(self.engine.get_stresses('колесом'), [5])

    def test_get_word_syllables(self):
        self.assertEqual(self.engine.get_word_syllables("корова"), ["ко", "ро", "ва"])

    def test_count_syllables(self):
        self.assertEqual(self.engine.count_syllables("корова"), 3)

    def test_is_rhyme(self):
        self.assertTrue(self.engine.is_rhyme("корова", "здорова"))

    def test_get_markup(self):
        self.assertIsInstance(self.engine.get_markup("корова"), Markup)

    def test_get_improved_markup(self):
        self.assertIsInstance(self.engine.get_improved_markup("корова")[0], Markup)

    def test_classify_metre(self):
        text = "Горит восток зарёю новой.\n" \
               "Уж на равнине, по холмам\n" \
               "Грохочут пушки. Дым багровый\n" \
               "Кругами всходит к небесам."
        self.assertEqual(self.engine.classify_metre(text), "iambos")

    def test_generate_poem(self):
        random.seed(42)
        model_path = GENERATOR_MODEL_DIR
        vocab_path = GENERATOR_WORD_FORM_VOCAB_PATH
        stress_path = GENERATOR_VOCAB_PATH
        poem = self.engine.generate_poem(
            model_path,
            vocab_path,
            stress_path,
            sampling_k=10000,
            n_syllables=8,
            rhyme_pattern="abab",
            metre_schema="-+")
        self.assertIsNotNone(poem)

    def test_get_word_rhymes(self):
        vocab_dump_file = os.path.join(EXAMPLES_DIR, "vocab_rhymes.pickle")
        self.assertEqual(self.engine.get_word_rhymes("глядел", vocab_dump_file, MARKUP_XML_EXAMPLE), ["сидел", "летел"])

