# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для выравнивания g2pц.

import unittest
import logging
import sys

from rupo.g2p.aligner import Aligner
from rupo.settings import RU_G2P_DICT_PATH


class TestAligner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    def test_aligner(self):
        aligner = Aligner()
        self.assertEqual(aligner.align('абазия', 'ɐbɐzʲijə'), ('абаз и я', 'ɐbɐzʲijə'))
        self.assertEqual(aligner.align('аахенец', 'aəxʲɪnʲɪʦ'), ('аах ен ец', 'aəxʲɪnʲɪʦ'))
        self.assertEqual(aligner.align('абатский', 'ɐbaʦkʲɪj'), ('абатск ий', 'ɐbaʦ kʲɪj'))
        self.assertEqual(aligner.align('абазинско-русский', 'ɐbɐzʲinskəruskʲɪj'),
                         ('абаз инско-русск ий', 'ɐbɐzʲinskə rus kʲɪj'))
        with open(RU_G2P_DICT_PATH, 'r', encoding='utf-8') as r:
            lines = r.readlines()[:50]
            pairs = [tuple(line.strip().split("\t")) for line in lines]
            for g, p in pairs:
                logging.debug(aligner.align(g, p))

