# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для классификатора g2p.

import unittest

from rupo.main.phoneme_classifier import MLPhonemeClassifier


class TestPhonemeClassifier(unittest.TestCase):
    def test_phoneme_classifier(self):
        self.assertEqual(MLPhonemeClassifier.align_phonemes("юла", "jʊɫa"), (' юла', 'jʊɫa'))
        self.assertEqual(MLPhonemeClassifier.align_phonemes("абажурчик", "ɐbɐʐurtɕɪk"), ('абажур чик', 'ɐbɐʐurtɕɪk'))
        print(MLPhonemeClassifier.generate_g2p_samples(" юла", "jʊɫa")[0][0])