from typing import List
from enum import Enum, unique, auto

from rupo.settings import HYPHEN_TOKENS
from rupo.main.markup import Annotation


class Token(Annotation):
    @unique
    class TokenType(Enum):
        """

        """
        UNKNOWN = -1
        WORD = auto()
        PUNCTUATION = auto()
        SPACE = auto()
        ENDLINE = auto()

        def __str__(self):
            return str(self.name)

        def __repr__(self):
            return self.__str__()

    def __init__(self, text: str, token_type: TokenType, begin: int, end: int):
        """
        :param text:
        :param token_type:
        :param begin:
        :param end:
        """
        self.token_type = token_type
        super(Token, self).__init__(begin, end, text)

    def __str__(self):
        return "'" + self.text + "'" + "|" + str(self.token_type) + " (" + str(self.begin) + ", " + str(self.end) + ")"

    def __repr__(self):
        return self.__str__()


class Tokenizer(object):
    @staticmethod
    def tokenize(text: str) -> List[Token]:
        """

        :param text:
        :return:
        """
        tokens = []
        punctuation = ".,?:;!—"
        begin = -1
        for i in range(len(text)):
            if text[i].isalpha() or text[i] == "-":
                if begin == -1:
                    begin = i
            else:
                if begin != -1:
                    tokens.append(Token(text[begin:i], Token.TokenType.WORD, begin, i))
                    begin = -1
                token_type = Token.TokenType.UNKNOWN
                if text[i] in punctuation:
                    token_type = Token.TokenType.PUNCTUATION
                elif text[i] == "\n":
                    token_type = Token.TokenType.ENDLINE
                elif text[i] == " ":
                    token_type = Token.TokenType.SPACE
                tokens.append(Token(text[i], token_type, i, i + 1))
        if begin != -1:
            tokens.append(Token(text[begin:len(text)], Token.TokenType.WORD, begin, len(text)))
        tokens = Tokenizer.__hyphen_map(tokens)
        return tokens

    @staticmethod
    def __hyphen_map(tokens: List[Token]) -> List[Token]:
        """

        :param tokens:
        :return:
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

        :return:
        """
        with open(HYPHEN_TOKENS, "r", encoding="utf-8") as file:
            hyphen_tokens = [token.strip() for token in file.readlines()]
            return hyphen_tokens
