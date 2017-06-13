# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для выравнивания g2pц.

import unittest

from rupo.g2p.aligner import Aligner
from rupo.settings import RU_G2P_DICT_PATH


class TestAligner(unittest.TestCase):
    def test_aligner(self):
        aligner = Aligner()
        with open(RU_G2P_DICT_PATH, 'r', encoding='utf-8') as r:
            lines = r.readlines()[:50]
            pairs = [tuple(line.strip().split("\t")) for line in lines]
            for g, p in pairs:
                print(aligner.align(g, p))

