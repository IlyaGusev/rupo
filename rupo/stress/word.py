# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Класс слова с ударением.

from enum import Enum
from typing import List, Set
from russ.syllables import get_syllables


class Stress:
    """
    Ударение
    """

    class Type(Enum):
        ANY = -1
        PRIMARY = 0
        SECONDARY = 1

    def __init__(self, position: int, stress_type: Type=Type.PRIMARY) -> None:
        self.position = position
        self.type = stress_type

    def __hash__(self):
        return hash(self.position)

    def __eq__(self, other: 'Stress'):
        return self.position == other.position and self.type == other.type

    def __str__(self):
        return str(self.position) + "\t" + str(self.type)

    def __repr__(self):
        return self.__str__()


class StressedWord:
    """
    Слово и его ударения.
    """

    def __init__(self, text: str, stresses: Set[Stress]) -> None:
        self.stresses = stresses
        self.text = text
        self.syllables = get_syllables(text)
        self.__accent_syllables()

    def get_primary_stresses(self) -> List[int]:
        return [stress.position for stress in self.stresses if stress.type == Stress.Type.PRIMARY]

    def get_secondary_stresses(self) -> List[int]:
        return [stress.position for stress in self.stresses if stress.type == Stress.Type.SECONDARY]

    def add_stress(self, position: int, stress_type: Stress.Type=Stress.Type.PRIMARY) -> None:
        self.stresses.add(Stress(position, stress_type))
        self.__accent_syllables()

    def add_stresses(self, stresses: List[Stress]) -> None:
        self.stresses = set(self.stresses).union(set(stresses))
        self.__accent_syllables()

    def __accent_syllables(self):
        for syllable in self.syllables:
            if Stress(syllable.vowel()) in self.stresses:
                syllable.stress = syllable.vowel()
            else:
                syllable.stress = -1

    def __str__(self):
        return self.text + "\t" + ",".join([str(i) for i in self.get_primary_stresses()])+ \
               "\t" + ",".join([str(i) for i in self.get_secondary_stresses()])

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.text)

    def __eq__(self, other: 'StressedWord'):
        return self.text == other.text
