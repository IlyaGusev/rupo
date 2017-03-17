# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты марковских цепей.

import unittest
import os

from rupo.settings import DICT_PATH, CLASSIFIER_PATH, MARKUPS_DUMP_XML_PATH, MARKOV_PICKLE
from rupo.generate.markov import MarkovModelContainer
from rupo.generate.generator import Generator
from rupo.accents.classifier import MLAccentClassifier
from rupo.accents.dict import AccentDict


class TestMarkovChains(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accents_dict = AccentDict(DICT_PATH)
        cls.accents_classifier = MLAccentClassifier(CLASSIFIER_PATH, cls.accents_dict)

    def test_generate(self):
        if os.path.exists(MARKUPS_DUMP_XML_PATH):
            markov = MarkovModelContainer(MARKOV_PICKLE, MARKUPS_DUMP_XML_PATH)
            generator = Generator(markov, markov.vocabulary)
            poem1 = generator.generate_poem(metre_schema="-+", rhyme_pattern="abab", n_syllables=8)
            self.assertEqual(len(poem1.split("\n")), 5)
            print(poem1)
            poem2 = generator.generate_poem(metre_schema="-+", rhyme_pattern="abab", n_syllables=8)
            self.assertEqual(len(poem2.split("\n")), 5)
            print(poem2)
            poem3 = generator.generate_poem(metre_schema="-+", rhyme_pattern="abba", n_syllables=8)
            self.assertEqual(len(poem3.split("\n")), 5)
            print(poem3)
            poem4 = generator.generate_poem(metre_schema="-+", rhyme_pattern="ababcc", n_syllables=10)
            self.assertEqual(len(poem4.split("\n")), 7)
            print(poem4)
            poem5 = generator.generate_poem_by_line(self.accents_dict, self.accents_classifier,
                                                    "Забывши волнения жизни мятежной,")
            print(poem5)
            poem6 = generator.generate_poem_by_line(self.accents_dict, self.accents_classifier,
                                                    "Просто первая строка")
            print(poem6)


