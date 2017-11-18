# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль токенизации.

import re
from typing import List
from enum import Enum, unique

from rupo.settings import HYPHEN_TOKENS


class Token:
    @unique
    class TokenType(Enum):
        """
        Тип токена.
        """
        UNKNOWN = -1
        WORD = 0
        PUNCTUATION = 1
        SPACE = 2
        ENDLINE = 3
        NUMBER = 4

        def __str__(self):
            return str(self.name)

        def __repr__(self):
            return self.__str__()

    def __init__(self, text: str, token_type: TokenType, begin: int, end: int):
        """
        :param text: исходный текст.
        :param token_type: тип токена.
        :param begin: начало позиции токена в тексте.
        :param end: конец позиции токена в тексте.
        """
        self.token_type = token_type
        self.begin = begin
        self.end = end
        self.text = text

    def __str__(self):
        return "'" + self.text + "'" + "|" + str(self.token_type) + " (" + str(self.begin) + ", " + str(self.end) + ")"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.text == other.text and self.token_type == other.token_type


class Tokenizer(object):
    """
    Класс токенизации.
    """
    @staticmethod
    def tokenize(text: str, remove_punct=False, remove_unknown=False, replace_numbers=False) -> List[Token]:
        """
        Токенизация текстов на русском языке с учётом знаков препинания и слов с дефисами.

        :param text: исходный текст.
        :return: список токенов.
        """
        tokens = []
        punctuation = ".,?:;!—"
        begin = -1
        for i, ch in enumerate(text):
            if ch.isalpha() or ch == "-":
                if begin == -1:
                    begin = i
            else:
                if begin != -1:
                    tokens.append(Tokenizer.__form_token(text, begin, i))
                    begin = -1
                token_type = Token.TokenType.UNKNOWN
                if ch in punctuation:
                    token_type = Token.TokenType.PUNCTUATION
                elif ch == "\n":
                    token_type = Token.TokenType.ENDLINE
                elif ch == " ":
                    token_type = Token.TokenType.SPACE
                elif ch.isdigit():
                    token_type = Token.TokenType.NUMBER
                if len(tokens) != 0 and tokens[-1].token_type == token_type:
                    tokens[-1].text += ch
                    tokens[-1].end += 1
                else:
                    tokens.append(Token(ch, token_type, i, i + 1))
        if begin != -1:
            tokens.append(Tokenizer.__form_token(text, begin, len(text)))
        tokens = Tokenizer.__hyphen_map(tokens)
        if remove_punct:
            tokens = [token for token in tokens if token.token_type != Token.TokenType.PUNCTUATION]
        if remove_unknown:
            tokens = [token for token in tokens if token.token_type != Token.TokenType.UNKNOWN]
        if replace_numbers:
            for token in tokens:
                if token.token_type != Token.TokenType.NUMBER:
                    continue
                token.text = "ЧИСЛО"
                token.token_type = Token.TokenType.WORD
        return tokens

    @staticmethod
    def __form_token(text, begin, end):
        word = text[begin:end]
        if word != "-":
            return Token(word, Token.TokenType.WORD, begin, end)
        else:
            return Token("-", Token.TokenType.PUNCTUATION, begin, begin + 1)

    @staticmethod
    def __hyphen_map(tokens: List[Token]) -> List[Token]:
        """
        Слова из словаря оставляем с дефисом, остальные разделяем.

        :param tokens: токены.
        :return: токены после обработки.
        """
        new_tokens = []
        hyphen_tokens = Tokenizer.__get_hyphen_tokens()
        for token in tokens:
            if token.token_type != Token.TokenType.WORD:
                new_tokens.append(token)
                continue
            is_one_word = True
            if "-" in token.text:
                is_one_word = False
                for hyphen_token in hyphen_tokens:
                    if hyphen_token in token.text or token.text in hyphen_token:
                        is_one_word = True
            if is_one_word:
                new_tokens.append(token)
            else:
                texts = token.text.split("-")
                pos = token.begin
                for text in texts:
                    new_tokens.append(Token(text, Token.TokenType.WORD, pos, pos+len(text)))
                    pos += len(text) + 1
        return new_tokens

    @staticmethod
    def __get_hyphen_tokens():
        """
        :return: содержание словаря, в котором прописаны слова с дефисом.
        """
        with open(HYPHEN_TOKENS, "r", encoding="utf-8") as file:
            hyphen_tokens = [token.strip() for token in file.readlines()]
            return hyphen_tokens


class SentenceTokenizer(object):
    @staticmethod
    def tokenize(text: str) -> List[str]:
        m = re.split(r'(?<=[^А-ЯЁ].[^А-ЯЁ][.?!;]) +(?=[А-ЯЁ])', text)
        return m
