# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Классификатор метра.

from collections import OrderedDict
from typing import List, Dict, Tuple
import jsonpickle

from rupo.main.markup import Line, Markup
from rupo.util.mixins import CommonMixin
from rupo.metre.pattern_analyzer import PatternAnalyzer
from rupo.util.preprocess import get_first_vowel_position


class StressCorrection(CommonMixin):
    """
    Исправление ударения.
    """
    def __init__(self, line_number: int, word_number: int, syllable_number: int,
                 word_text: str, stress: int) -> None:
        """
        :param line_number: номер строки.
        :param word_number: номер слова.
        :param syllable_number: номер слога.
        :param word_text: текст слова.
        :param stress: позиция ударения (с 0).
        """
        self.line_number = line_number
        self.word_number = word_number
        self.syllable_number = syllable_number
        self.word_text = word_text
        self.accent = stress


class ClassificationResult(CommonMixin):
    """
    Результат классификации стихотворения по метру.
    """
    def __init__(self, count_lines: int=0) -> None:
        """
        :param count_lines: количество строк.
        """
        self.metre = None
        self.count_lines = count_lines
        self.errors_count = {k: 0 for k in MetreClassifier.metres.keys()}  # type: Dict[str, int]
        self.corrections = {k: [] for k in MetreClassifier.metres.keys()}  # type: Dict[str, List[StressCorrection]]
        self.resolutions = {k: [] for k in MetreClassifier.metres.keys()}  # type: Dict[str, List[StressCorrection]]
        self.additions = {k: [] for k in MetreClassifier.metres.keys()}  # type: Dict[str, List[StressCorrection]]

    def get_metre_errors_count(self):
        """
        :return: получить количество ошибок на заданном метре.
        """
        return self.errors_count[self.metre]

    def to_json(self):
        """
        :return: сериализация в json.
        """
        return jsonpickle.encode(self)

    @staticmethod
    def str_corrections(collection: List[StressCorrection]) -> str:
        """
        :param collection: список исправлений.
        :return: его строковое представление.
        """
        return"\n".join([str((item.word_text, item.syllable_number)) for item in collection])

    def __str__(self):
        st = "Метр: " + str(self.metre) + "\n"
        st += "Снятая омография: \n" + ClassificationResult.str_corrections(self.resolutions[self.metre]) + "\n"
        st += "Неправильные ударения: \n" + ClassificationResult.str_corrections(self.corrections[self.metre]) + "\n"
        st += "Новые ударения: \n" + ClassificationResult.str_corrections(self.additions[self.metre]) + "\n"
        return st


class ErrorsTableRecord:
    def __init__(self, strong_errors, weak_errors, pattern):
        self.strong_errors = strong_errors
        self.weak_errors = weak_errors
        self.pattern = pattern

    def __str__(self):
        return self.pattern + " " + str(self.strong_errors) + " " + str(self.weak_errors)

    def __repr__(self):
        return self.__str__()


class ErrorsTable:
    def __init__(self, num_lines):
        self.data = {}
        self.num_lines = num_lines
        self.coef = OrderedDict(
            [("iambos", 0.3),
             ("choreios", 0.3),
             ("daktylos", 0.6),
             ("amphibrachys", 0.6),
             ("anapaistos", 0.6),
             ("dolnik3", 2.4),
             ("dolnik2", 2.4),
             ("taktovik3", 5.0),
             ("taktovik2", 5.0)
             ])
        self.sum_coef = OrderedDict(
            [("iambos", 0.0),
             ("choreios", 0.0),
             ("daktylos", 0.0),
             ("amphibrachys", 0.0),
             ("anapaistos", 0.0),
             ("dolnik3", 0.02),
             ("dolnik2", 0.02),
             ("taktovik3", 0.1),
             ("taktovik2", 0.1)
             ])
        for metre_name in MetreClassifier.metres.keys():
            self.data[metre_name] = [(0, 0) for _ in range(num_lines)]

    def add_record(self, metre_name, line_num, strong_errors, weak_errors, pattern):
        self.data[metre_name][line_num] = ErrorsTableRecord(strong_errors, weak_errors, pattern)

    def get_best_metre(self):
        for l in range(self.num_lines):
            strong_sum = 0
            weak_sum = 0
            for metre_name in self.data.keys():
                strong_sum += self.data[metre_name][l].strong_errors
                weak_sum += self.data[metre_name][l].weak_errors
            for metre_name, column in self.data.items():
                if strong_sum != 0:
                    column[l].strong_errors = column[l].strong_errors / float(strong_sum)
                if weak_sum != 0:
                    column[l].weak_errors = column[l].weak_errors / float(weak_sum)
        sums = dict()
        for metre_name in self.data.keys():
            sums[metre_name] = (0, 0)
        for metre_name, column in self.data.items():
            strong_sum = 0
            weak_sum = 0
            for l in range(self.num_lines):
                strong_sum += column[l].strong_errors
                weak_sum += column[l].weak_errors
            sums[metre_name] = (strong_sum, weak_sum)
        for metre_name, pair in sums.items():
            sums[metre_name] = self.sum_coef[metre_name] + (pair[0] + pair[1] / 2.0) * self.coef[metre_name] / self.num_lines
        print(sums)
        return min(sums, key=sums.get)


class MetreClassifier(object):
    """
    Классификатор, считает отклонения от стандартных шаблонов ритма(метров).
    """
    metres = OrderedDict(
        [("iambos", '(us)*(uS)(U)?(U)?'),
         ("choreios", '(su)*(S)(U)?(U)?'),
         ("daktylos", '(suu)*(S)(U)?(U)?'),
         ("amphibrachys", '(usu)*(uS)(U)?(U)?'),
         ("anapaistos",  '(uus)*(uuS)(U)?(U)?'),
         ("dolnik3", '(u)?(u)?((su)(u)?)*(S)(U)?(U)?'),
         ("dolnik2", '(u)?(u)?((s)(u)?)*(S)(U)?(U)?'),
         ("taktovik3", '(u)?(u)?((su)(u)?(u)?)*(S)(U)?(U)?'),
         ("taktovik2", '(u)?(u)?((s)(u)?(u)?)*(S)(U)?(U)?')
         ])

    border_syllables_count = 18

    @staticmethod
    def classify_metre(markup):
        """
        Классифицируем стихотворный метр.

        :param markup: разметка.
        :return: результат классификации.
        """
        result = ClassificationResult(len(markup.lines))
        num_lines = len(markup.lines)
        errors_table = ErrorsTable(num_lines)
        for l, line in enumerate(markup.lines):
            for metre_name, metre_pattern in MetreClassifier.metres.items():
                line_syllables_count = sum([len(word.syllables) for word in line.words])

                # Строчки длиной больше border_syllables_count слогов не обрабатываем.
                if line_syllables_count > MetreClassifier.border_syllables_count or line_syllables_count == 0:
                    continue
                pattern, strong_errors, weak_errors = \
                    PatternAnalyzer.count_errors(MetreClassifier.metres[metre_name],
                                                 MetreClassifier.__get_line_pattern(line))
                if len(pattern) == 0:
                    errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern)
                    continue
                corrections = MetreClassifier.__get_line_pattern_matching_corrections(line, l, pattern)[0]
                accentuation_errors = len(corrections)
                strong_errors += accentuation_errors
                errors_table.add_record(metre_name, l, strong_errors, weak_errors, pattern)
        result.metre = errors_table.get_best_metre()

        # Запомним все исправления.
        for l, line in enumerate(markup.lines):
            pattern = errors_table.data[result.metre][l].pattern
            if len(pattern) == 0:
                continue
            corrections, resolutions, additions =\
                MetreClassifier.__get_line_pattern_matching_corrections(line, l, pattern)
            result.corrections[result.metre] += corrections
            result.resolutions[result.metre] += resolutions
            result.additions[result.metre] += additions
            result.errors_count[result.metre] += len(corrections)
        return result

    @staticmethod
    def __get_line_pattern(line: Line) -> str:
        """
        Сопоставляем строку шаблону, считаем ошибки.

        :param line: строка.
        :return: количество ошибок
        """
        pattern = ""
        for w, word in enumerate(line.words):
            if len(word.syllables) == 0:
                pattern += "U"
            else:
                for syllable in word.syllables:
                    if syllable.accent != -1:
                        pattern += "S"
                    else:
                        pattern += "U"
        return pattern

    @staticmethod
    def __get_line_pattern_matching_corrections(line: Line, line_number: int, pattern: str) \
            -> Tuple[List[StressCorrection], List[StressCorrection], List[StressCorrection]]:
        """
        Ударения могут приходиться на слабое место,
        если безударный слог того же слова не попадает на икт. Иначе - ошибка.

        :param line: строка.
        :param line_number: номер строки.
        :param pattern: шаблон.
        :return: ошибки, дополнения и снятия
        """
        corrections = []
        resolutions = []
        additions = []
        number_in_pattern = 0
        for w, word in enumerate(line.words):
            # Игнорируем слова длиной меньше 2 слогов.
            if len(word.syllables) <= 1:
                number_in_pattern += len(word.syllables)
                continue
            stress_count = word.count_stresses()
            for syllable in word.syllables:
                if stress_count == 0 and pattern[number_in_pattern].lower() == "s":
                    # Ударений нет, ставим такое, какое подходит по метру. Возможно несколько.
                    additions.append(StressCorrection(line_number, w, syllable.number, word.text, syllable.vowel()))
                elif pattern[number_in_pattern] == "u" and syllable.accent != -1:
                    # Ударение есть и оно падает на этот слог, при этом в шаблоне безударная позиция.
                    # Найдём такой слог, у которого в шаблоне ударная позиция. Это и есть наше исправление.
                    for other_syllable in word.syllables:
                        other_number_in_pattern = other_syllable.number - syllable.number + number_in_pattern
                        if syllable.number == other_syllable.number or pattern[other_number_in_pattern].lower() != "s":
                            continue
                        ac = StressCorrection(line_number, w, other_syllable.number, word.text, other_syllable.vowel())
                        if stress_count == 1 and other_syllable.accent == -1:
                            corrections.append(ac)
                        else:
                            resolutions.append(ac)
                number_in_pattern += 1
        return corrections, resolutions, additions

    @staticmethod
    def get_improved_markup(markup: Markup, result: ClassificationResult) -> Markup:
        """
        Улучшаем разметку после классификации метра.

        :param markup: начальная разметка.
        :param result: результат классификации.
        :return: улучшенная разметка.
        """
        for pos in result.corrections[result.metre] + result.resolutions[result.metre]:
            syllables = markup.lines[pos.line_number].words[pos.word_number].syllables
            for i, syllable in enumerate(syllables):
                syllable.stress = -1
                if syllable.number == pos.syllable_number:
                    syllable.stress = syllable.begin + get_first_vowel_position(syllable.text)

        for pos in result.additions[result.metre]:
            syllable = markup.lines[pos.line_number].words[pos.word_number].syllables[pos.syllable_number]
            syllable.stress = syllable.begin + get_first_vowel_position(syllable.text)

        return markup

    @staticmethod
    def improve_markup(markup: Markup) -> \
            Tuple[Markup, ClassificationResult]:
        """
        Улучшение разметки метрическим классификатором.

        :param markup: начальная разметка.
        """
        result = MetreClassifier.classify_metre(markup)
        return MetreClassifier.get_improved_markup(markup, result), result
