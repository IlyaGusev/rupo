# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Сопоставление шаблону.

from typing import List, Set, Tuple


class TreeNode:
    """
    Нода дерева разбора шаблона.
    """
    leaf_chars = "usUS"
    non_leaf_chars = "*?w"

    def __init__(self, parent: 'TreeNode', children: List['TreeNode'], text: str, pattern_pos: int):
        """
        :param parent: родитель ноды.
        :param children: дети ноды.
        :param text: символ, соответствующий ноде.
        :param pattern_pos: позиция символа в шаблоне
        """
        self.parent = parent  # type: TreeNode
        self.children = children  # type: List[TreeNode]
        self.text = text  # type: str
        self.pattern_pos = pattern_pos  # type: int

    def get_level(self) -> int:
        """
        :return: высота ноды в дереве.
        """
        parent = self.parent
        level = 0
        while parent is not None:
            parent = parent.parent
            level += 1
        return level

    def get_next_sibling(self) -> 'TreeNode':
        """
        :return: соседняя нода справа.
        """
        siblings = self.parent.children
        index = siblings.index(self) + 1
        if index < len(siblings):
            return siblings[index]
        return None

    def get_last_child_leaf(self) -> 'TreeNode':
        """
        :return: последний лист из детей.
        """
        for child in reversed(self.children):
            if child.is_leaf():
                return child
        return None

    def print_tree(self) -> None:
        """
        Вывод дерева с корнем в этой ноде.
        """
        stack = list()
        stack.append(self)
        while len(stack) != 0:
            current_node = stack.pop()
            print("\t" * current_node.get_level(), current_node)
            stack += current_node.children

    def is_leaf(self) -> bool:
        """
        :return: является ли нода листом дерева.
        """
        return self.text in TreeNode.leaf_chars

    def __str__(self) -> str:
        return self.text + " " + str(self.pattern_pos)

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self):
        return hash(self.pattern_pos)

    def __eq__(self, other):
        return self.pattern_pos == other.pattern_pos


class State:
    """
    Состояние разбора.
    """
    def __init__(self, node: TreeNode, string_pos: int, strong_errors: int, weak_errors: int, pattern: str):
        """
        :param node: нода дерева, соответствующая состоянию.
        :param string_pos: позиция в сопоставляемой строке.
        :param strong_errors: количество ошибок в U и S.
        :param weak_errors: количество ошибок в u и s.
        :param pattern: шаблон - путь, до этого состояния.
        """
        self.node = node  # type: TreeNode
        self.string_pos = string_pos  # type: int
        self.strong_errors = strong_errors  # type: int
        self.weak_errors = weak_errors  # type: int
        self.pattern = pattern  # type: str

    def __str__(self) -> str:
        return str(self.node) + " " + str(self.string_pos) + " " + str(self.strong_errors) + " " + str(self.weak_errors)

    def __repr__(self) -> str:
        return self.__str__()


class PatternAnalyzer:
    """
    Сопоставлятель шаблона и строки.
    """
    error_border = 10

    def __init__(self, pattern):
        """
        :param pattern: шаблон.
        """
        self.pattern = pattern  # type: str
        self.tree = self.__build_tree(pattern)  # type: TreeNode

    @staticmethod
    def count_errors(pattern: str, string: str) -> Tuple[str, int, int]:
        """
        :param pattern: шаблон.
        :param string: строка.
        :return: лучший шаблон, количество сильных ошибок, количество слабых ошибок.
        """
        analyzer = PatternAnalyzer(pattern)
        return analyzer.__accept(string)

    def __accept(self, string: str) -> Tuple[str, int, int]:
        """
        :param string: строка.
        :return: лучший шаблон, количество сильных ошибок, количество слабых ошибок.
        """
        current_states = [State(None, -1, 0, 0, "")]
        current_node = self.__get_most_left_leaf(self.tree)
        for i, ch in enumerate(string):
            new_states = []
            for state in current_states:
                if state.node is not None:
                    current_node = self.__get_next_leaf(state.node)
                variants = self.__get_variants(current_node)

                # Каждый вариант - новое состояние.
                for variant in variants:
                    assert variant.is_leaf()
                    strong_errors = state.strong_errors + int(variant.text.isupper() and variant.text != ch)
                    weak_errors = state.weak_errors + int(variant.text.islower() and variant.text != ch.lower())
                    new_state = State(variant, i, strong_errors, weak_errors, state.pattern+variant.text)
                    if new_state.strong_errors + new_state.weak_errors > PatternAnalyzer.error_border:
                        continue
                    new_states.append(new_state)

            if len(new_states) == 0:
                # Можем закончить раньше, если по ошибкам порезали ветки, либо если шаблон меньше строки.
                current_states = PatternAnalyzer.__filter_states(current_states, self.tree)
                pattern, strong_errors, weak_errors = self.__get_min_errors_from_states(current_states)
                diff = (len(string) - i)
                return pattern, strong_errors + diff, weak_errors + diff

            current_states = new_states
        current_states = PatternAnalyzer.__filter_states(current_states, self.tree)
        return self.__get_min_errors_from_states(current_states)

    @staticmethod
    def __filter_states(states: List[State], root: TreeNode):
        """
        Фильтрация по наличию обязательных терминалов.
        
        :param states: состояния.
        :param root: корень дерева.
        :return: отфильтрованные состояния.
        """
        return [state for state in states if root.get_last_child_leaf() is None or
                state.node.pattern_pos >= root.get_last_child_leaf().pattern_pos]

    @staticmethod
    def __build_tree(pattern: str) -> TreeNode:
        """
        Построение дерева шаблона.
        
        :param pattern: шаблон.
        :return: корень дерева.
        """
        root_node = TreeNode(None, list(), "R", -1)
        current_node = root_node
        for i, ch in enumerate(pattern):
            if ch == "(":
                node = TreeNode(current_node, list(), "()", i)
                current_node.children.append(node)
                current_node = node
            if ch == ")":
                node = current_node
                current_node = current_node.parent
                # Убираем бессмысленные скобки.
                if i + 1 < len(pattern) and pattern[i + 1] not in "*?":
                    current_node.children = current_node.children[:-1] + node.children
                    for child in node.children:
                        child.parent = current_node
            if ch in TreeNode.leaf_chars:
                current_node.children.append(TreeNode(current_node, list(), ch, i))
            # Заменяем скобки на нетерминалы.
            if ch in TreeNode.non_leaf_chars:
                current_node.children[-1].text = ch
                current_node.children[-1].pattern_pos = i
        return root_node

    @staticmethod
    def __get_min_errors_from_states(states) -> Tuple[str, int, int]:
        """
        :param states: состояния.
        :return: лучший шаблон, количество сильных ошибок, количество слабых ошибок.
        """
        if len(states) == 0:
            return "", 0, 0
        return min([(state.pattern, state.strong_errors, state.weak_errors) for i, state in enumerate(states)],
                   key=lambda x: (x[1], x[2], x[0]))

    @staticmethod
    def __get_most_left_leaf(node: TreeNode) -> TreeNode:
        """
        Самый левый потомок.
        
        :param node: текущая нода.
        :return: самый левый потомок.
        """
        while len(node.children) != 0:
            node = node.children[0]
        return node

    @staticmethod
    def __get_variants(current_node) -> Set[TreeNode]:
        """
        :param current_node: текущая нода.
        :return: варианты ноды на том же символе строки, возникают из-за * и ? в шаблоне.
        """
        variants = set()
        current_variant = current_node
        while current_variant is not None:
            if current_variant not in variants:
                variants.add(current_variant)
            else:
                current_variant = current_variant.parent
            current_variant = PatternAnalyzer.__get_next_variant(current_variant)
        return variants

    @staticmethod
    def __get_next_variant(node):
        """
        Получение следующего варианта.
        
        :param node: текущий вариант.
        :return: следующий вариант. 
        """
        assert node.is_leaf()
        while node.parent is not None:
            parent = node.parent
            grandfather = parent.parent
            uncle = parent.get_next_sibling() if grandfather is not None else None
            if (not node.is_leaf() or PatternAnalyzer.__is_first_leaf(node)) and uncle is not None:
                return PatternAnalyzer.__get_most_left_leaf(uncle)
            elif grandfather is not None and grandfather.text == "*" and grandfather.children[-1] == parent:
                return PatternAnalyzer.__get_most_left_leaf(grandfather)
            if parent.children[0] == node:
                node = parent
            else:
                return None
        return None

    @staticmethod
    def __is_first_leaf(node) -> bool:
        leaves = [child for child in node.parent.children if child.is_leaf()]
        if node not in leaves:
            return False
        return leaves.index(node) == 0

    @staticmethod
    def __get_next_leaf(node):
        while node.parent is not None:
            sibling = node.get_next_sibling()
            if sibling is not None:
                return PatternAnalyzer.__get_most_left_leaf(sibling)
            elif node.parent.text == "*" and node.parent.children[-1] == node:
                return PatternAnalyzer.__get_most_left_leaf(node.parent.children[0])
            node = node.parent
        return None