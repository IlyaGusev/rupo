# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для токенизатора.

import unittest

from rupo.main.tokenizer import Tokenizer, Token


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        text1 = "О, когда-нибудь, когда?"
        self.assertEqual(Tokenizer.tokenize(text1), [
            Token('О', Token.TokenType.WORD, 0, 1),
            Token(',', Token.TokenType.PUNCTUATION, 1, 2),
            Token(' ', Token.TokenType.SPACE, 2, 3),
            Token('когда-нибудь', Token.TokenType.WORD, 3, 15),
            Token(',', Token.TokenType.PUNCTUATION, 15, 16),
            Token(' ', Token.TokenType.SPACE, 16, 17),
            Token('когда', Token.TokenType.WORD, 17, 22),
            Token('?', Token.TokenType.PUNCTUATION, 22, 23)])
