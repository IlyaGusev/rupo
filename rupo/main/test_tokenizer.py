# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для токенизатора.

import unittest

from rupo.main.tokenizer import Tokenizer, SentenceTokenizer, Token


class TestTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        text = "О, когда-нибудь, когда?"
        self.assertEqual(Tokenizer.tokenize(text), [
            Token('О', Token.TokenType.WORD, 0, 1),
            Token(',', Token.TokenType.PUNCTUATION, 1, 2),
            Token(' ', Token.TokenType.SPACE, 2, 3),
            Token('когда-нибудь', Token.TokenType.WORD, 3, 15),
            Token(',', Token.TokenType.PUNCTUATION, 15, 16),
            Token(' ', Token.TokenType.SPACE, 16, 17),
            Token('когда', Token.TokenType.WORD, 17, 22),
            Token('?', Token.TokenType.PUNCTUATION, 22, 23)])

        text = " Пора"
        self.assertEqual(Tokenizer.tokenize(text), [
            Token(' ', Token.TokenType.SPACE, 0, 1),
            Token('Пора', Token.TokenType.WORD, 1, 5)])

    def test_numbers(self):
        text = "Очевидно, 1 января 1970 года..."
        self.assertEqual(Tokenizer.tokenize(text), [
            Token('Очевидно', Token.TokenType.WORD, 0, 8),
            Token(',', Token.TokenType.PUNCTUATION, 8, 9),
            Token(' ', Token.TokenType.SPACE, 9, 10),
            Token('1', Token.TokenType.NUMBER, 10, 11),
            Token(' ', Token.TokenType.SPACE, 11, 12),
            Token('января', Token.TokenType.WORD, 12, 18),
            Token(' ', Token.TokenType.SPACE, 18, 19),
            Token('1970', Token.TokenType.NUMBER, 19, 23),
            Token(' ', Token.TokenType.SPACE, 23, 24),
            Token('года', Token.TokenType.WORD, 24, 28),
            Token('...', Token.TokenType.PUNCTUATION, 28, 31)])

        self.assertEqual(Tokenizer.tokenize(text, replace_numbers=True), [
            Token('Очевидно', Token.TokenType.WORD, 0, 8),
            Token(',', Token.TokenType.PUNCTUATION, 8, 9),
            Token(' ', Token.TokenType.SPACE, 9, 10),
            Token('ЧИСЛО', Token.TokenType.WORD, 10, 11),
            Token(' ', Token.TokenType.SPACE, 11, 12),
            Token('января', Token.TokenType.WORD, 12, 18),
            Token(' ', Token.TokenType.SPACE, 18, 19),
            Token('ЧИСЛО', Token.TokenType.WORD, 19, 23),
            Token(' ', Token.TokenType.SPACE, 23, 24),
            Token('года', Token.TokenType.WORD, 24, 28),
            Token('...', Token.TokenType.PUNCTUATION, 28, 31)])


class TestSentenceTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        text1 = "Конкурс учреждён в 2005 году!!! Официальный партнёр конкурса – Президентский центр Б.Н. Ельцина."
        self.assertEqual(SentenceTokenizer.tokenize(text1),
                         ['Конкурс учреждён в 2005 году!!!',
                          'Официальный партнёр конкурса – Президентский центр Б.Н. Ельцина.'])
