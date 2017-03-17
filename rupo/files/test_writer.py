# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты записи разметок.

import unittest
import os

from rupo.files.writer import Writer
from rupo.files.reader import Reader, FileTypeEnum
from rupo.util.data import MARKUP_EXAMPLE
from rupo.settings import EXAMPLES_DIR


class TestWriter(unittest.TestCase):
    def test_write(self):
        temp_file = os.path.join(EXAMPLES_DIR, "temp.xml")
        markup = MARKUP_EXAMPLE
        Writer.write_markups(FileTypeEnum.XML, [markup], temp_file)
        processed_xml = Reader.read_markups(temp_file, FileTypeEnum.XML, is_processed=True)
        self.assertEqual(next(processed_xml), markup)
        processed_xml.close()
        os.remove(temp_file)
