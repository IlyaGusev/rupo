# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты записи разметок.

import unittest
import os

from rupo.main.markup import Markup
from rupo.files.writer import Writer
from rupo.files.reader import Reader, FileType
from rupo.util.data import MARKUP_EXAMPLE
from rupo.settings import EXAMPLES_DIR


class TestWriter(unittest.TestCase):
    def test_write(self):
        temp_file = os.path.join(EXAMPLES_DIR, "temp.xml")
        markup = MARKUP_EXAMPLE
        Writer.write_markups(FileType.XML, [markup], temp_file)
        processed_xml = Reader.read_markups(temp_file, FileType.XML, is_processed=True)
        self.assertEqual(next(processed_xml), markup)
        processed_xml.close()
        os.remove(temp_file)

        temp_file = os.path.join(EXAMPLES_DIR, "temp.txt")
        Writer.write_markups(FileType.RAW, [markup], temp_file)
        processed_raw = Reader.read_markups(temp_file, FileType.RAW, is_processed=True)
        self.assertIsInstance((next(processed_raw)), Markup)
        processed_raw.close()
        os.remove(temp_file)
