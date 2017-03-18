# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты к компилятору выражений.

import unittest

from rupo.metre.patterns import Patterns


class TestPatterns(unittest.TestCase):
    def test_patterns(self):
        self.assertCountEqual(Patterns.compile_pattern("(s)*", 3), ["sss"])
        self.assertCountEqual(Patterns.compile_pattern("(s)?(u)?(S)?", 2), ["su", "uS", "sS"])
        self.assertCountEqual(Patterns.compile_pattern("(s)?(u)(s)*", 1), ["u"])
        self.assertCountEqual(Patterns.compile_pattern("(s)?(u)(s)*", 2), ["su", "us"])
        self.assertCountEqual(Patterns.compile_pattern("(s)?(u)(s)*", 3), ["sus", "uss"])
        self.assertCountEqual(Patterns.compile_pattern("(us)*(uS)(U)?(U)?", 4), ["usuS", "uSUU"])
        self.assertCountEqual(Patterns.compile_pattern("((s)(u)?)*", 4), ["susu", "ssss", "suss", 'sssu', 'ssus'])


