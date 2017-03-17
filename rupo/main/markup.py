# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль для описания разметки по ударениям и слогам.

import json
from typing import List
import xml.etree.ElementTree as etree

from dicttoxml import dicttoxml

from rupo.util.preprocess import get_first_vowel_position
from rupo.util.mixins import CommonMixin


class Annotation(CommonMixin):
    """
    Класс аннотации.
    Содержит начальную и конечную позицию в тексте, а также текст аннотации .
    """
    def __init__(self, begin: int, end: int, text: str) -> None:
        self.begin = begin
        self.end = end
        self.text = text


class Syllable(Annotation):
    """
    Разметка слога. Включает в себя аннотацию и номер слога, а также ударение.
    Если ударение падает не на этот слог, -1.
    """
    def __init__(self, begin: int, end: int, number: int, text: str, accent: int=-1) -> None:
        super(Syllable, self).__init__(begin, end, text)
        self.number = number
        self.accent = accent

    def vowel(self) -> int:
        """
        :return: позиция гласной буквы этого слога в слове (с 0).
        """
        return get_first_vowel_position(self.text) + self.begin

    def from_dict(self, d: dict) -> 'Syllable':
        self.__dict__.update(d)
        return self


class Word(Annotation):
    """
    Разметка слова. Включает в себя аннотацию слова и его слоги.
    """
    def __init__(self, begin: int, end: int, text: str, syllables: List[Syllable]) -> None:
        super(Word, self).__init__(begin, end, text)
        self.syllables = syllables

    def count_accents(self) -> int:
        """
        :return: количество ударений в слове.
        """
        return sum(syllable.accent != -1 for syllable in self.syllables)

    def accent(self) -> int:
        """
        :return: последнее ударение в слове, если нет, то -1.
        """
        accent = -1
        for syllable in self.syllables:
            if syllable.accent != -1:
                accent = syllable.accent
        return accent

    def get_accented_syllables_numbers(self) -> List[int]:
        """
        :return: номера слогов, на которые падают ударения.
        """
        return [syllable.number for syllable in self.syllables if syllable.accent != -1]

    def set_accents(self, accents: List[int]) -> None:
        """
        Задать ударения, все остальные убираются.

        :param accents: позиции ударения в слове.
        """
        for syllable in self.syllables:
            if syllable.vowel() in accents:
                syllable.accent = syllable.vowel()
            else:
                syllable.accent = -1

    def get_short(self) -> str:
        """
        :return: слово в форме "текст"+"последнее ударение".
        """
        return self.text.lower() + str(self.accent())

    def from_dict(self, d: dict) -> 'Word':
        self.__dict__.update(d)
        syllables = d["syllables"]  # type: List[dict]
        self.syllables = [Syllable(0, 0, 0, "").from_dict(syllables[i]) for i in range(len(syllables))]
        return self

    def __hash__(self) -> int:
        """
        :return: хеш разметки.
        """
        return hash(self.get_short())


class Line(Annotation):
    """
    Разметка строки. Включает в себя аннотацию строки и её слова.
    """
    def __init__(self, begin: int, end: int, text: str, words: List[Word]) -> None:
        super(Line, self).__init__(begin, end, text)
        self.words = words

    def from_dict(self, d) -> 'Line':
        self.__dict__.update(d)
        words = d["words"]  # type: List[dict]
        self.words = [Word(0, 0, "", []).from_dict(words[i]) for i in range(len(words))]
        return self


class Markup(CommonMixin):
    """
    Класс данных для разметки в целом с экспортом/импортом в XML и JSON.
    """
    def __init__(self, text: str=None, lines: List[Line]=None) -> None:
        # TODO: При изменении структуры разметки менять десериализацию.
        self.text = text
        self.lines = lines
        self.version = 2

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def from_json(self, st) -> 'Markup':
        d = json.loads(st)
        return self.from_dict(d)

    def from_dict(self, d) -> 'Markup':
        self.__dict__.update(d)
        lines = d["lines"]  # type: List[dict]
        self.lines = [Line(0, 0, "", []).from_dict(lines[i]) for i in range(len(lines))]
        return self

    def to_xml(self) -> str:
        """
        Экспорт в XML.

        :return self: строка в формате XML
        """
        return dicttoxml(self.to_dict(), custom_root='markup', attr_type=False).decode('utf-8').replace("\n", "\\n")

    def from_xml(self, xml: str) -> 'Markup':
        """
        Импорт из XML.

        :param xml: XML-разметка
        :return self: получившийся объект Markup
        """
        root = etree.fromstring(xml)
        if root.find("version") is None or int(root.find("version").text) != self.version:
            raise TypeError("Другая версия разметки")
        lines_node = root.find("lines")
        lines = []
        for line_node in lines_node.findall("item"):
            words_node = line_node.find("words")
            words = []
            for word_node in words_node.findall("item"):
                syllables_node = word_node.find("syllables")
                syllables = []
                for syllable_node in syllables_node.findall("item"):
                    syllables.append(Syllable(int(syllable_node.find("begin").text),
                                              int(syllable_node.find("end").text),
                                              int(syllable_node.find("number").text),
                                              syllable_node.find("text").text,
                                              int(syllable_node.find("accent").text)))
                words.append(Word(int(word_node.find("begin").text), int(word_node.find("end").text),
                                  word_node.find("text").text, syllables))
            lines.append(Line(int(line_node.find("begin").text), int(line_node.find("end").text),
                              line_node.find("text").text, words))
        self.text = root.find("text").text.replace("\\n", "\n")
        self.lines = lines
        return self
