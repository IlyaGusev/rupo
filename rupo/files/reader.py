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
from rupo.accents.dict import AccentDict
from rupo.accents.classifier import MLAccentClassifier
from rupo.metre.metre_classifier import MetreClassifier


RAW_SEPARATOR = "\n\n\n"


class FileTypeEnum(Enum):
    """
    Тип файла.
    """
    RAW = ".txt"
    XML = ".xml"
    JSON = ".json"
    STIHI = ""


class Reader(object):
    """
    Считывание из файлов.
    """
    @staticmethod
    def read_markups(path: str, source_type: FileTypeEnum, is_processed: bool,
                     accents_dict: AccentDict=None, accents_classifier: MLAccentClassifier=None) -> Iterator[Markup]:
        """
        Считывание разметок (включая разметку по сырым текстам).

        :param path: путь к файлу/папке.
        :param source_type: тип файлов.
        :param is_processed: уже размеченные тексты?
        :param accents_dict: словарь ударений (для неразмеченных текстов).
        :param accents_classifier: классификатор ударений (для неразмеченных текстов).
        """
        paths = Reader.__get_paths(path, source_type.value)
        for filename in paths:
            with open(filename, "r", encoding="utf-8") as file:
                if is_processed:
                    if source_type == FileTypeEnum.XML:
                        for elem in Reader.__xml_iter(file, 'markup'):
                            markup = Markup()
                            markup.from_xml(etree.tostring(elem, encoding='utf-8', method='xml'))
                            yield markup
                    elif source_type == FileTypeEnum.JSON:
                        j = json.load(file)
                        for item in j['items']:
                            markup = Markup()
                            markup.from_dict(item)
                            yield markup
                    elif source_type == FileTypeEnum.RAW:
                        raise NotImplementedError("Пока не реализовано.")
                else:
                    assert accents_dict is not None
                    assert accents_classifier is not None
                    for text in Reader.read_texts(filename, source_type):
                        yield Reader.__markup_text(text, accents_dict, accents_classifier)

    @staticmethod
    def read_texts(path: str, source_type: FileTypeEnum) -> Iterator[str]:
        """
        Считывание текстов.

        :param path: путь к файлу/папке.
        :param source_type: тип файлов.
        """
        paths = Reader.__get_paths(path, source_type.value)
        for filename in paths:
            with open(filename, "r", encoding="utf-8") as file:
                if source_type == FileTypeEnum.XML:
                    for elem in Reader.__xml_iter(file, 'item'):
                        yield elem.find(".//text").text
                elif source_type == FileTypeEnum.JSON:
                    # TODO: ленивый парсинг
                    j = json.load(file)
                    for item in j['items']:
                        yield item['text']
                elif source_type == FileTypeEnum.RAW:
                    text = file.read()
                    for t in text.split(RAW_SEPARATOR):
                        yield t
                elif source_type == FileTypeEnum.STIHI:
                    text = ""
                    is_text = False
                    for line in file:
                        if "<div" in line:
                            is_text = True
                        elif "</div>" in line:
                            is_text = False
                            yield text
                        elif is_text:
                            text += line + "\n"

    @staticmethod
    def __get_paths(path: str, ext: str) -> Iterator[str]:
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
                    return Reader.__get_paths(folder, ext)

    @staticmethod
    def __markup_text(text: str, accents_dict: AccentDict=None,
                      accents_classifier: MLAccentClassifier=None) -> Markup:
        """
        Разметка текста.

        :param text: текст.
        :param accents_dict: словарь ударений.
        :param accents_classifier: классификатор ударений.
        :return: разметка.
        """
        markup = Phonetics.process_text(text, accents_dict)
        markup = MetreClassifier.improve_markup(markup, accents_classifier)[0]
        return markup

    @staticmethod
    def __xml_iter(file, tag):
        """
        :param file: xml файл.
        :param tag: заданный тег.
        :return: все элементы с заданными тегами в xml.
        """
        return (elem for event, elem in etree.iterparse(file, events=['end']) if event == 'end' and elem.tag == tag)
