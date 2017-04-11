# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для классификатора g2p.

import unittest

from rupo.main.phoneme_classifier import MLPhonemeClassifier


class TestPhonemeClassifier(unittest.TestCase):
    def test_phoneme_classifier(self):
        clf = MLPhonemeClassifier()
        print(clf.predict("ведь"))
        print(clf.predict("здравствуйте"))
        print(clf.predict("корова"))
        print(clf.predict("скрупулёзный"))
        print(clf.predict("палец"))
        print(clf.predict("черёмуха"))
        print(clf.predict("цирк"))