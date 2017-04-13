# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты марковских цепей.

import os
import unittest

from rupo.generate.markov import MarkovModelContainer
from rupo.main.vocabulary import Vocabulary
from rupo.settings import EXAMPLES_DIR, MARKUP_XML_EXAMPLE


class TestMarkov(unittest.TestCase):
    def test_markov(self):
        for n in range(2, 5):
            vocab_dump_file = os.path.join(EXAMPLES_DIR, "vocab.pickle")
            markov_dump_file = os.path.join(EXAMPLES_DIR, "markov.pickle")
            vocabulary = Vocabulary(vocab_dump_file, MARKUP_XML_EXAMPLE)
            markov = MarkovModelContainer(markov_dump_file, vocabulary, MARKUP_XML_EXAMPLE, n_grams=n)
            self.assertTrue(os.path.exists(vocab_dump_file))
            self.assertTrue(os.path.exists(markov_dump_file))
            os.remove(vocab_dump_file)
            os.remove(markov_dump_file)
            self.assertEqual(vocabulary.size()-n+1, len(markov.transitions))
            self.assertEqual(sum([sum(transition.values()) for transition in markov.transitions.values()]), vocabulary.size()-n+1)