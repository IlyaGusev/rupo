# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Шаблоны ритма.

from collections import defaultdict
from typing import List, Dict
from enum import Enum


class FlagEnum(Enum):
    ZERO_OR_ONE = "?"
    ZERO_TO_INF = "*"
    ONE = ""


class Group:
    """
    Группа - выражение в круглых скобках и её модификатор.
    """
    def __init__(self, exp: str, flag: FlagEnum) -> None:
        """
        :param exp: выражение в круглых скобках.
        :param flag: модификатор.
        """
        self.exp = exp
        self.flag = flag
        self.extended = []  # type: List[str]

    def __str__(self):
        return self.exp + str(self.flag) + ": " + str(self.extended)

    def __repr__(self):
        return self.__str__()


class Patterns:
    @staticmethod
    def compile_pattern(metre_pattern: str, l: int) -> List[str]:
        """
        Скомпилировать заданное выражение.

        :param metre_pattern: выражение.
        :param l: итоговое количество символов.
        :return: скомпилированные шаблоны.
        """
        group = Group(metre_pattern, FlagEnum.ONE)
        Patterns.__process_group(group, l)
        return list(set([st for st in group.extended if len(st) == l]))

    @staticmethod
    def __process_group(group: Group, l: int) -> None:
        """
        Обработка группы. Записывает в extended результаты.

        :param group: группа.
        :param l: максимальное количество символов.
        """
        # Высчитываем группы следующего уровня.
        children_groups = Patterns.__find_groups(group.exp)
        if len(children_groups) == 0:
            # Нижний уровень.
            if group.flag == FlagEnum.ZERO_OR_ONE or group.flag == FlagEnum.ONE:
                group.extended = [group.exp]
            elif group.flag == FlagEnum.ZERO_TO_INF:
                group.extended = [group.exp * i for i in range(1, l // len(group.exp) + 1)]
            else:
                assert False
        else:
            # Спускаемся.
            for child in children_groups:
                Patterns.__process_group(child, l)
            # Поднимаемся, параллельно проставляем extended
            group.extended = Patterns.__next(children_groups, 0, "", l)

    @staticmethod
    def __find_groups(expression: str) -> List[Group]:
        """
        Высчитываем группы выражения.

        :param expression: выражение.
        :return: группы.
        """
        groups = []
        counter = 0
        begin = -1
        for i in range(len(expression)):
            if expression[i] == "(":
                counter += 1
                if counter == 1:
                    begin = i + 1
            if expression[i] == ")":
                if counter == 1:
                    flag = FlagEnum.ONE  # type: FlagEnum
                    if i + 1 < len(expression) and (expression[i + 1] == "?" or expression[i + 1] == "*"):
                        flag = FlagEnum(expression[i + 1])
                    groups.append(Group(expression[begin:i], flag))
                counter -= 1
            assert counter >= 0
        assert counter == 0
        return groups

    @staticmethod
    def __next(groups: List[Group], index: int, pattern: str, l: int) -> List[str]:
        """
        Получение всех возможных шаблонов на заданных группах заданной длины.

        :param groups: группы.
        :param index: индекс текущей группы в массиве.
        :param pattern: текущий набранный шаблон
        :param l: максимальное количество символов.
        :return: все возможные шаблоны.
        """
        # База рекурсии.
        if len(pattern) > l or index >= len(groups):
            if sum(groups[i].flag == FlagEnum.ONE for i in range(index, len(groups))) == 0:
                return [pattern, ]
            return []
        result = []
        flag = groups[index].flag
        if flag == FlagEnum.ZERO_TO_INF:
            # Продолжаем все варианты *
            for i in range(len(groups[index].extended)):
                result += Patterns.__next(groups, index, pattern + groups[index].extended[i], l)
            # Пропускаем *
            result += Patterns.__next(groups, index + 1, pattern, l)
        elif flag == FlagEnum.ZERO_OR_ONE:
            # Пропускаем ?
            result += Patterns.__next(groups, index + 1, pattern, l)
            # Продолжаем ?
            for st in groups[index].extended:
                result += Patterns.__next(groups, index + 1, pattern + st, l)
        elif flag == FlagEnum.ONE:
            for st in groups[index].extended:
                result += Patterns.__next(groups, index + 1, pattern + st, l)
        return result


class CompiledPatterns(object):
    """
    Скомпилированные шаблоны (нужно, чтобы не пересчитывать их каждый раз).
    """
    def __init__(self, metres: Dict[str, str], border: int) -> None:
        self.compilations = defaultdict(lambda: defaultdict(lambda: ""))
        for metre_name, metre_pattern in metres.items():
            for i in range(1, border+1):
                self.compilations[metre_name][i] = Patterns.compile_pattern(metre_pattern, i)

    def get_patterns(self, metre_name: str, syllables_count: int) -> List[str]:
        """
        Получить или посчитать шаблоны метра для заданного метра.

        :param metre_name: название метра.
        :param syllables_count: количество слогов в шаблоне.
        :return: итоговые скомпилированные шаблоны метра.
        """
        return self.compilations[metre_name][syllables_count]
