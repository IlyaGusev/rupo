# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Запись в файлы разных расширений.

import os
from typing import List

from rupo.files.reader import RAW_SEPARATOR
from rupo.main.markup import Markup
from rupo.files.reader import FileTypeEnum


class Writer(object):
    """
    Запись в файл.
    """
    def __init__(self, destination_type: FileTypeEnum, path: str) -> None:
        """
        Нужно, когда хотим записывать разметки по одной (экономия памяти).

        :param destination_type: тип файла.
        :param path: путь к файлу.
        """
        self.type = destination_type
        self.path = path
        self.file = None
        try:
            os.remove(path)
        except OSError:
            pass

    def open(self) -> None:
        """
        Открываем файл, вызывать до начала записи.
        """
        self.file = open(self.path, "w", encoding="utf-8")
        if self.type == FileTypeEnum.XML:
            self.file.write('<?xml version="1W.0" encoding="UTF-8"?><items>')

    def write_markup(self, markup: Markup) -> None:
        """
        Запись разметки в уже открытый файл.
        :param markup: разметка.
        """
        assert self.file is not None
        if self.type == FileTypeEnum.XML:
            xml = markup.to_xml().encode('utf-8')\
                .replace(b'<?xml version="1.0" encoding="UTF-8" ?>', b'').decode('utf-8')
            self.file.write(xml)
        elif self.type == FileTypeEnum.RAW:
            Writer.__write_markup_raw(markup, self.file)

    def close(self) -> None:
        """
        Закрываем файл.
        """
        if self.type == FileTypeEnum.XML:
            self.file.write('</items>')
        self.file.close()

    @staticmethod
    def write_markups(destination_type: FileTypeEnum, markups: List[Markup], path: str) -> None:
        """
        Запись разметок в файл.

        :param destination_type: тип файла.
        :param markups: разметки.
        :param path: путь к файлу.
        """
        with open(path, "w", encoding="utf-8") as file:
            if destination_type == FileTypeEnum.XML:
                file.write('<?xml version="1.0" encoding="UTF-8"?><items>')
                for markup in markups:
                    xml = markup.to_xml().encode('utf-8')\
                        .replace(b'<?xml version="1.0" encoding="UTF-8" ?>', b'').decode('utf-8')
                    file.write(xml)
                    file.write("\n")
                file.write('</items>')
            elif destination_type == FileTypeEnum.JSON:
                file.write("[")
                for markup in markups:
                    file.write(markup.to_json())
                    file.write(",")
                file.seek(0, 2)
                size = file.tell()
                file.truncate(size - 1)
                file.write(']')
            elif destination_type == FileTypeEnum.RAW:
                for markup in markups:
                    Writer.__write_markup_raw(markup, file)

    @staticmethod
    def __write_markup_raw(markup: Markup, file) -> None:
        """
        Запись разметки в текстовом виде (слово+ударение).

        :param markup: разметка.
        :param file: открытый файл.
        """
        lines = []
        for line in markup.lines:
            lines.append(" ".join([word.get_short() for word in line.words]))
        file.write("\n".join(lines))
        file.write(RAW_SEPARATOR)
