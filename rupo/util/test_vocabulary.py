# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты словаря.

import unittest
import os

from rupo.settings import EXAMPLES_DIR, MARKUP_XML_EXAMPLE
from rupo.util.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):
    def test_vocabulary(self):
        dump_file = os.path.join(EXAMPLES_DIR, "temp.pickle")
        vocabulary = Vocabulary(dump_file, MARKUP_XML_EXAMPLE)
        self.assertTrue(os.path.exists(dump_file))
        os.remove(dump_file)
        try:
            self.assertTrue(vocabulary.get_word(0) is not None)
        except IndexError:
            self.assertTrue(False)
