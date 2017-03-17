# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты записи разметок.

import unittest
import os

from rupo.convertion.writer import Writer
from rupo.convertion.reader import Reader, FileTypeEnum
from rupo.accents.classifier import MLAccentClassifier
from rupo.accents.dict import AccentDict
from rupo.util.data import MARKUP_EXAMPLE


# class TestWriter(unittest.TestCase):
#     @classmethod
#     def setUpClass(cls):
#         cls.accents_dict = AccentDict()
#         cls.accents_classifier = MLAccentClassifier(cls.accents_dict)
#
#     def test_write(self):
#         tempfile = os.path.join(BASE_DIR, "datasets", "temp.xml")
#         markup = MARKUP_EXAMPLE
#         Writer.write_markups(FileTypeEnum.XML, [markup], tempfile)
#         processed_xml = Reader.read_markups(tempfile, FileTypeEnum.XML, is_processed=True)
#         self.assertEqual(next(processed_xml), markup)
#         processed_xml.close()
#         os.remove(tempfile)
