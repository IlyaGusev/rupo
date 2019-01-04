# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты словаря.

import os
import unittest

from rupo.main.vocabulary import StressVocabulary
from rupo.settings import EXAMPLES_DIR, MARKUP_XML_EXAMPLE


class TestVocabulary(unittest.TestCase):
    def test_vocabulary(self):
        dump_file = os.path.join(EXAMPLES_DIR, "temp.pickle")
        vocabulary = StressVocabulary()
        vocabulary.parse(MARKUP_XML_EXAMPLE)
        vocabulary.save(dump_file)
        self.assertTrue(os.path.exists(dump_file))
        os.remove(dump_file)
        try:
            self.assertTrue(vocabulary.get_word(0) is not None)
        except IndexError:
            self.assertTrue(False)
