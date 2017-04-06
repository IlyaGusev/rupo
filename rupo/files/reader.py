# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Считыватель файлов разных расширений.

import os
import xml.etree.ElementTree as etree
import json
from enum import Enum
from typing import Iterator

from rupo.main.markup import Markup
from rupo.main.phonetics import Phonetics
from rupo.stress.dict import StressDict
from rupo.stress.classifier import MLStressClassifier
from rupo.metre.metre_classifier import MetreClassifier


RAW_SEPARATOR = "\n\n\n"


class FileType(Enum):
    """
    Тип файла.
    """
    RAW = ".txt"
    XML = ".xml"
    JSON = ".json"
    VOCAB = ".voc"


class Reader(object):
    """
    Считывание из файлов.
    """
    @staticmethod
    def read_markups(path: str, source_type: FileType, is_processed: bool,
                     stress_dict: StressDict=None, stress_classifier: MLStressClassifier=None) -> Iterator[Markup]:
        """
        Считывание разметок (включая разметку по сырым текстам).

        :param path: путь к файлу/папке.
        :param source_type: тип файлов.
        :param is_processed: уже размеченные тексты?
        :param stress_dict: словарь ударений (для неразмеченных текстов).
        :param stress_classifier: классификатор ударений (для неразмеченных текстов).
        """
        paths = Reader.get_paths(path, source_type.value)
        for filename in paths:
            with open(filename, "r", encoding="utf-8") as file:
                if is_processed:
                    if source_type == FileType.XML:
                        for elem in Reader.__xml_iter(file, 'markup'):
                            markup = Markup()
                            markup.from_xml(etree.tostring(elem, encoding='utf-8', method='xml'))
                            yield markup
                    elif source_type == FileType.JSON:
                        j = json.load(file)
                        for item in j['items']:
                            markup = Markup()
                            markup.from_dict(item)
                            yield markup
                    elif source_type == FileType.RAW:
                        separator_count = 0
                        text = ""
                        for line in file:
                            if line == "\n":
                                separator_count += 1
                            else:
                                text += line
                            if separator_count == 3:
                                separator_count = 0
                                markup = Markup()
                                markup.from_raw(text)
                                yield markup
                        if text != "":
                            markup = Markup()
                            markup.from_raw(text)
                            yield markup
                else:
                    assert stress_dict is not None
                    assert stress_classifier is not None
                    for text in Reader.read_texts(filename, source_type):
                        yield Reader.__markup_text(text, stress_dict, stress_classifier)

    @staticmethod
    def read_vocabulary(path: str):
        """
        Считывание словаря.

        :param path: путь к словарю.
        :return: слово и его индекс.
        """
        paths = Reader.get_paths(path, FileType.VOCAB.value)
        for filename in paths:
            with open(filename, "r", encoding="utf-8") as file:
                for line in file:
                    markup = Markup()
                    fields = line.strip().split('\t')
                    markup.from_raw(fields[0])
                    yield markup.lines[0].words[0], int(fields[1])

    @staticmethod
    def read_texts(path: str, source_type: FileType) -> Iterator[str]:
        """
        Считывание текстов.

        :param path: путь к файлу/папке.
        :param source_type: тип файлов.
        """
        paths = Reader.get_paths(path, source_type.value)
        for filename in paths:
            with open(filename, "r", encoding="utf-8") as file:
                if source_type == FileType.XML:
                    for elem in Reader.__xml_iter(file, 'item'):
                        yield elem.find(".//text").text
                elif source_type == FileType.JSON:
                    # TODO: ленивый парсинг
                    j = json.load(file)
                    for item in j['items']:
                        yield item['text']
                elif source_type == FileType.RAW:
                    text = file.read()
                    for t in text.split(RAW_SEPARATOR):
                        yield t

    @staticmethod
    def get_paths(path: str, ext: str) -> Iterator[str]:
        """
        Получение всех файлов заданного типа по заданному пути.

        :param path: путь к файлу/папке.
        :param ext: требуемое расширение.
        """
        if os.path.isfile(path):
            if ext == os.path.splitext(path)[1]:
                yield path
        else:
            for root, folders, files in os.walk(path):
                for file in files:
                    if ext == os.path.splitext(file)[1]:
                        yield os.path.join(root, file)
                for folder in folders:
                    return Reader.get_paths(folder, ext)

    @staticmethod
    def __markup_text(text: str, stress_dict: StressDict=None,
                      stress_classifier: MLStressClassifier=None) -> Markup:
        """
        Разметка текста.

        :param text: текст.
        :param stress_dict: словарь ударений.
        :param stress_classifier: классификатор ударений.
        :return: разметка.
        """
        markup = Phonetics.process_text(text, stress_dict)
        markup = MetreClassifier.improve_markup(markup, stress_classifier)[0]
        return markup

    @staticmethod
    def __xml_iter(file, tag):
        """
        :param file: xml файл.
        :param tag: заданный тег.
        :return: все элементы с заданными тегами в xml.
        """
        return (elem for event, elem in etree.iterparse(file, events=['end']) if event == 'end' and elem.tag == tag)
