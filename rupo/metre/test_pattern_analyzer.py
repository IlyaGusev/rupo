# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты к компилятору выражений.

import unittest

from rupo.metre.pattern_analyzer import PatternAnalyzer


class TestPatternAnalyzer(unittest.TestCase):
    def test_pattern_analyzer(self):
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "uuu"), ('sss', 0, 3, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "uus"), ('sss', 0, 2, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "usu"), ('sss', 0, 2, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "uss"), ('sss', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "suu"), ('sss', 0, 2, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "sus"), ('sss', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "ssu"), ('sss', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)*", "sss"), ('sss', 0, 0, False))

        self.assertEqual(PatternAnalyzer.count_errors("(sus)*(u)?", 'suu'), ('sus', 0, 1, False))

        self.assertEqual(PatternAnalyzer.count_errors("((sus)*u)*s", 'susss'), ('susus', 0, 1, False))

        self.assertEqual(PatternAnalyzer.count_errors("(s((s)*u)*)*", 'susss'), ('susss', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s((s)*u)*)*", 'usss'), ('ssss', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s((s)*u)*)*", 'suuu'), ('suuu', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s((s)*u)*)*", 'suuusuuus'), ('suuusuuus', 0, 0, False))

        self.assertEqual(PatternAnalyzer.count_errors("(sss((sus)*uss)*)*", 'ssssussususs'), ('ssssussususs', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(sss((sus)*uss)*)*", 'ssssuuuss'), ('ssssususs', 0, 1, False))

        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "uuuu"), ('susu', 0, 2, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "uuus"), ('ssus', 0, 2, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "uusu"), ('susu', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "uuss"), ('suss', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "usuu"), ('sssu', 0, 2, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "usus"), ('ssus', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "ussu"), ('sssu', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "usss"), ('ssss', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "suuu"), ('susu', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "suus"), ('ssus', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "susu"), ('susu', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "suss"), ('suss', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "ssuu"), ('sssu', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", 'ssus'), ('ssus', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", 'sssu'), ('sssu', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("((s)(u)?)*", "ssss"), ('ssss', 0, 0, False))

        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)?(S)?", "su"), ('su', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)?(S)?", "ss"), ('su', 0, 1, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)?(S)?", "uS"), ('uS', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)?(S)?", "sS"), ('sS', 0, 0, False))

        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)(s)*", "u"), ('u', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)(s)*", "su"), ('su', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)(s)*", "us"), ('us', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)(s)*", "sus"), ('sus', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(s)?(u)(s)*", "uss"), ('uss', 0, 0, False))

        self.assertEqual(PatternAnalyzer.count_errors("(us)*(uS)(U)?(U)?", "usuS"), ('usuS', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(us)*(uS)(U)?(U)?", "uSUU"), ('uSUU', 0, 0, False))

        self.assertEqual(PatternAnalyzer.count_errors("(su(u)?)*", "su"), ('su', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(su(u)?)*", "suu"), ('suu', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(su(u)?)*", "susu"), ('susu', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(su(u)?)*", "suusuu"), ('suusuu', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(su(u)?)*", "ssussu"), ('suusuu', 0, 2, False))

        self.assertEqual(PatternAnalyzer.count_errors("(u)?(u)?((s)(u)?(u)?)*(S)(U)?(U)?", "sssuSU"), ('sssuSU', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(u)?(u)?((s)(u)?(u)?)*(S)(U)?(U)?", "ussuSU"), ('ussuSU', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(u)?(u)?((s)(u)?(u)?)*(S)(U)?(U)?", "susuuSU"), ('susuuSU', 0, 0, False))
        self.assertEqual(PatternAnalyzer.count_errors("(u)?(u)?((s)(u)?(u)?)*(S)(U)?(U)?", "uusuuSU"), ('uusuuSU', 0, 0, False))
