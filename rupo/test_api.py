# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для API библиотеки.

import unittest
import os

from rupo.settings import MARKUP_XML_EXAMPLE, EXAMPLES_DIR
from rupo.main.markup import Markup
from rupo.api import get_accent, get_word_syllables, count_syllables, get_markup, get_improved_markup, \
    is_rhyme, classify_metre, generate_poem, get_word_rhymes, Global


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
        self.assertIsInstance(get_improved_markup("корова")[0], Markup)

    def test_classify_metre(self):
        text = "Горит восток зарёю новой.\n" \
               "Уж на равнине, по холмам\n" \
               "Грохочут пушки. Дым багровый\n" \
               "Кругами всходит к небесам."
        self.assertEqual(classify_metre(text), "iambos")

    def test_generate_poem(self):
        vocab_dump_file = os.path.join(EXAMPLES_DIR, "vocab.pickle")
        markov_dump_file = os.path.join(EXAMPLES_DIR, "markov.pickle")
        self.assertNotEqual(
            generate_poem(MARKUP_XML_EXAMPLE, markov_dump_file, vocab_dump_file, rhyme_pattern="aa", n_syllables=6), "")
        os.remove(vocab_dump_file)
        os.remove(markov_dump_file)
        Global.vocabulary = None
        Global.markov = None
        Global.generator = None

    def test_get_word_rhymes(self):
        vocab_dump_file = os.path.join(EXAMPLES_DIR, "vocab.pickle")
        self.assertEqual(get_word_rhymes("глядел", vocab_dump_file, MARKUP_XML_EXAMPLE), ["сидел"])
        os.remove(vocab_dump_file)
