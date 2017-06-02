# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для API библиотеки.

import unittest
import os

from rupo.settings import MARKUP_XML_EXAMPLE, EXAMPLES_DIR
from rupo.main.markup import Markup
from rupo.api import Engine


class TestApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine(language="ru")
        cls.engine.load()

    def test_stress(self):
        self.assertEqual(self.engine.get_stress("корова"), 3)
        self.assertEqual(self.engine.get_stress("когда-нибудь"), 9)

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
        vocab_dump_file = os.path.join("/home/yallen/Документы/Python/rupo", "vocab.pickle")
        markov_dump_file = os.path.join("/home/yallen/Документы/Python/rupo", "markov.pickle")
        self.assertIsNotNone(self.engine.generate_poem("/home/yallen/Документы/Python/Poems/datasets/corpus/markup_dump.xml", markov_dump_file,
                                                       vocab_dump_file, rhyme_pattern="aa", n_syllables=6))
        print(self.engine.generate_poem("/home/yallen/Документы/Python/Poems/datasets/corpus/markup_dump.xml", markov_dump_file,
                                        vocab_dump_file, rhyme_pattern="aabb", n_syllables=8, beam_width=10))
        print(self.engine.generate_poem("/home/yallen/Документы/Python/Poems/datasets/corpus/markup_dump.xml",
                                        markov_dump_file,
                                        vocab_dump_file, rhyme_pattern="aabb", n_syllables=8, beam_width=10))
        print(self.engine.generate_poem("/home/yallen/Документы/Python/Poems/datasets/corpus/markup_dump.xml",
                                        markov_dump_file,
                                        vocab_dump_file, rhyme_pattern="aabb", n_syllables=8, beam_width=10))
        os.remove(vocab_dump_file)
        os.remove(markov_dump_file)
        self.engine.vocabulary = None
        self.engine.markov = None
        self.engine.generator = None

    def test_get_word_rhymes(self):
        vocab_dump_file = os.path.join(EXAMPLES_DIR, "vocab.pickle")
        self.assertEqual(self.engine.get_word_rhymes("глядел", vocab_dump_file, MARKUP_XML_EXAMPLE), ["сидел", "летел"])
        os.remove(vocab_dump_file)
