# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль токенизации.

from typing import List
from enum import Enum, unique

from rupo.settings import HYPHEN_TOKENS
from rupo.main.markup import Annotation


class Token(Annotation):
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
        super(Token, self).__init__(begin, end, text)

    def __str__(self):
        return "'" + self.text + "'" + "|" + str(self.token_type) + " (" + str(self.begin) + ", " + str(self.end) + ")"

    def __repr__(self):
        return self.__str__()


class Tokenizer(object):
    """
    Класс токенизации.
    """
    @staticmethod
    def tokenize(text: str) -> List[Token]:
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
                    tokens.append(Token(text[begin:i], Token.TokenType.WORD, begin, i))
                    begin = -1
                token_type = Token.TokenType.UNKNOWN
                if ch in punctuation:
                    token_type = Token.TokenType.PUNCTUATION
                elif ch == "\n":
                    token_type = Token.TokenType.ENDLINE
                elif ch == " ":
                    token_type = Token.TokenType.SPACE
                tokens.append(Token(ch, token_type, i, i + 1))
        if begin != -1:
            tokens.append(Token(text[begin:len(text)], Token.TokenType.WORD, begin, len(text)))
        tokens = Tokenizer.__hyphen_map(tokens)
        return tokens

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
