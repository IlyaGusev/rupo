# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс для удобной работы со словарём ударений.

import pygtrie
import os
import pickle
from typing import List, Dict, ItemsView, Set

from rupo.dict.cmu import CMUDict
from rupo.settings import RU_GRAPHEME_STRESS_PATH, RU_GRAPHEME_STRESS_TRIE_PATH, \
    EN_PHONEME_STRESS_PATH, EN_PHONEME_STRESS_TRIE_PATH, ZALYZNYAK_DICT, CMU_DICT

from rupo.stress.word import Stress


class StressDict:
    """
    Класс данных, для сериализации словаря как префиксного дерева и быстрой загрузки в память.
    """

    class Mode:
        GRAPHEMES = 0
        PHONEMES = 0

    def __init__(self, language: str="ru", mode: Mode=Mode.GRAPHEMES, raw_dict_path=None, trie_path=None,
                 zalyzniak_dict=ZALYZNYAK_DICT, cmu_dict=CMU_DICT) -> None:
        self.data = pygtrie.Trie()  # type: Dict[str, Set[Stress]]
        self.raw_dict_path = raw_dict_path
        self.trie_path = trie_path
        if language == "ru" and mode == self.Mode.GRAPHEMES:
            self.__init_defaults(RU_GRAPHEME_STRESS_PATH, RU_GRAPHEME_STRESS_TRIE_PATH)
            if not os.path.exists(self.raw_dict_path):
                from rupo.dict.zaliznyak import ZalyzniakDict
                ZalyzniakDict.convert_to_accent_only(zalyzniak_dict, self.raw_dict_path)
        elif mode == self.Mode.PHONEMES and language == "en":
            self.__init_defaults(EN_PHONEME_STRESS_PATH, EN_PHONEME_STRESS_TRIE_PATH)
            if not os.path.exists(self.raw_dict_path):
                CMUDict.convert_to_phoneme_stress(cmu_dict, self.raw_dict_path)
        else:
            assert False
        if not os.path.isfile(self.raw_dict_path):
            raise FileNotFoundError("Dictionary raw file not found.")
        if os.path.isfile(self.trie_path):
            self.load(self.trie_path)
        else:
            self.create(self.raw_dict_path, self.trie_path)

    def __init_defaults(self, raw_dict_path, trie_path):
        if self.raw_dict_path is None:
            self.raw_dict_path = raw_dict_path
        if self.trie_path is None:
            self.trie_path = trie_path

    def create(self, src_filename: str, dst_filename: str) -> None:
        """
        Загрузка словаря из файла.

        :param src_filename: имя файла с оригинальным словарём.
        :param dst_filename: имя файла, в который будет сохранён дамп.
        """
        with open(src_filename, 'r', encoding='utf-8') as f:
            for line in f:
                word, primary, secondary = line.split("\t")
                stresses = [Stress(int(a), Stress.Type.PRIMARY) for a in primary.strip().split(",")]
                if secondary.strip() != "":
                    stresses += [Stress(int(a), Stress.Type.SECONDARY) for a in secondary.strip().split(",")]
                self.update(word, stresses)
        self.save(dst_filename)

    def save(self, dst_filename: str) -> None:
        """
        Сохранение дампа.
        
        :param dst_filename: имя файла, в который сохраняем дамп словаря.
        """
        with open(dst_filename, "wb") as f:
            pickle.dump(self.data, f, pickle.HIGHEST_PROTOCOL)

    def load(self, dump_filename: str) -> None:
        """
        Загрузка дампа словаря.
        
        :param dump_filename: откуда загружаем.
        """
        with open(dump_filename, "rb") as f:
            self.data = pickle.load(f)

    def get_stresses(self, word: str, stress_type: Stress.Type=Stress.Type.ANY) -> List[int]:
        """
        Получение ударений нужного типа у слова.

        :param word: слово, которое мы хотим посмотреть в словаре.
        :param stress_type: тип ударения.
        :return forms: массив всех ударений.
        """
        if word in self.data:
            if stress_type == Stress.Type.ANY:
                return [stress.position for stress in self.data[word]]
            else:
                return [stress.position for stress in self.data[word] if stress.type == stress_type]
        return []

    def get_all(self) -> ItemsView[str, Set[Stress]]:
        """
        :return items: все ключи и ударения словаря.
        """
        return self.data.items()

    def update(self, word: str, stresses: List[Stress]) -> None:
        """
        Обновление словаря.

        :param word: слово.
        :param stresses: набор ударений.
        """
        if word not in self.data:
            self.data[word] = set(stresses)
        else:
            self.data[word].update(stresses)

    def update_primary_only(self, word: str, stresses: List[int]) -> None:
        self.update(word, [Stress(stress, Stress.Type.PRIMARY) for stress in stresses])