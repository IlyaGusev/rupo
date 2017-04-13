# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс для удобной работы со словарём ударений.

import datrie
import os
from typing import List, Tuple
from enum import Enum

from rupo.util.preprocess import CYRRILIC_LOWER_VOWELS, CYRRILIC_LOWER_CONSONANTS
from rupo.settings import DICT_TXT_PATH
from rupo.settings import DICT_TRIE_PATH


class StressDict:
    """
    Класс данных, для сериализации словаря как префиксного дерева и быстрой загрузки в память.
    """

    class StressType(Enum):
        ANY = -1
        PRIMARY = 0
        SECONDARY = 1

    StressType = StressType

    def __init__(self) -> None:
        self.data = datrie.Trie(CYRRILIC_LOWER_VOWELS+CYRRILIC_LOWER_CONSONANTS+"-")
        src_filename = DICT_TXT_PATH
        dst_filename = DICT_TRIE_PATH
        if not os.path.isfile(src_filename):
            raise FileNotFoundError("Не найден файл словаря.")
        if os.path.isfile(dst_filename):
            self.data = datrie.Trie.load(dst_filename)
        else:
            self.create(src_filename, dst_filename)

    def create(self, src_filename: str, dst_filename: str) -> None:
        """
        Загрузка словаря из файла. Если уже есть его сериализация в .trie файле, берём из него.

        :param src_filename: имя файла с оригинальным словарём.
        :param dst_filename: имя файла, в который будет сохранён дамп.
        """
        with open(src_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                for word in line.split("#")[1].split(","):
                    word = word.strip()
                    pos = -1
                    stresses = []
                    clean_word = ""
                    for i, ch in enumerate(word):
                        if ch == "'" or ch == "`":
                            if ch == "`":
                                stresses.append((pos, StressDict.StressType.SECONDARY))
                            else:
                                stresses.append((pos, StressDict.StressType.PRIMARY))
                            continue
                        clean_word += ch
                        pos += 1
                        if ch == "ё":
                            stresses.append((pos, StressDict.StressType.PRIMARY))
                    self.update(clean_word, stresses)
        self.data.save(dst_filename)

    def save(self, dst_filename: str) -> None:
        """
        Сохранение дампа.

        :param dst_filename: имя файла, в который сохраняем дамп словаря.
        """
        self.data.save(dst_filename)

    def get_stresses(self, word: str, stress_type: StressType=StressType.ANY) -> List[int]:
        """
        Получение ударений нужного типа у слова.

        :param word: слово, которое мы хотим посмотреть в словаре.
        :param stress_type: тип ударения.
        :return forms: массив всех ударений.
        """
        if word in self.data:
            if stress_type == StressDict.StressType.ANY:
                return [i[0] for i in self.data[word]]
            else:
                return [i[0] for i in self.data[word] if i[1] == stress_type]
        return []

    def get_all(self) -> List[Tuple[str, List[Tuple[int, StressType]]]]:
        """
        :return items: все ключи и ударения словаря.
        """
        return self.data.items()

    def update(self, word: str, stress_pairs: List[Tuple[int, StressType]]) -> None:
        """
        Обновление словаря.

        :param word: слово.
        :param stress_pairs: набор ударений.
        """
        if word not in self.data:
            self.data[word] = set(stress_pairs)
        else:
            self.data[word].update(stress_pairs)
