# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для классификатора g2p.

import unittest
from rupo.g2p.decision import DecisionPhonemePredictor


class TestPhonemeClassifier(unittest.TestCase):
    def test_phoneme_classifier(self):
        clf = DecisionPhonemePredictor()
        print(clf.predict("ведь"))
        print(clf.predict("здравствуйте"))
        print(clf.predict("корова"))
        print(clf.predict("скрупулёзный"))
        print(clf.predict("палец"))
        print(clf.predict("черёмуха"))
        print(clf.predict("цирк"))

