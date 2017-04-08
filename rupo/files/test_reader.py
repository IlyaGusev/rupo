# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты считывателя разметок.

import unittest

from rupo.files.reader import Reader, FileType
from rupo.stress.stress_classifier import MLStressClassifier
from rupo.stress.dict import StressDict
from rupo.main.markup import Markup, Line, Word
from rupo.settings import MARKUP_XML_EXAMPLE, TEXT_XML_EXAMPLE, MARKUP_JSON_EXAMPLE


class TestReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress_dict = StressDict()
        cls.stress_classifier = MLStressClassifier(cls.stress_dict)

    def test_read(self):
        processed_xml = Reader.read_markups(MARKUP_XML_EXAMPLE, FileType.XML, is_processed=True)
        self.__assert_markup_is_correct(next(processed_xml))

        unprocessed_xml = Reader.read_markups(TEXT_XML_EXAMPLE, FileType.XML, is_processed=False,
                                              stress_dict=self.stress_dict,
                                              stress_classifier=self.stress_classifier)
        self.__assert_markup_is_correct(next(unprocessed_xml))

        processed_json = Reader.read_markups(MARKUP_JSON_EXAMPLE, FileType.JSON, is_processed=True)
        self.__assert_markup_is_correct(next(processed_json))

    def __assert_markup_is_correct(self, markup):
        self.assertIsInstance(markup, Markup)
        self.assertIsNotNone(markup.text)
        self.assertNotEqual(markup.text, "")
        self.assertNotEqual(markup.lines, [])
        self.assertIsInstance(markup.lines[0], Line)
        self.assertIsInstance(markup.lines[0].words[0], Word)
