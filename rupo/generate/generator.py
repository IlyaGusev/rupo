# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль создания стихотворений.

from typing import List
import copy

import numpy as np
from numpy.random import choice

from rupo.stress.predictor import StressPredictor
from rupo.generate.filters import MetreFilter, RhymeFilter
from rupo.main.vocabulary import Vocabulary
from rupo.main.markup import Markup
from rupo.metre.metre_classifier import MetreClassifier, CompilationsSingleton
from rupo.generate.model_container import ModelContainer
from rupo.generate.word_form_vocabulary import WordFormVocabulary


class BeamPath(object):
    """
    Путь в лучевом поиске.
    """
    def __init__(self, indices: List[int], metre_filter: MetreFilter, rhyme_filter: RhymeFilter,
                 probability: float, line_ends: List[int]):
        """
        :param indices: индексы слов на пути.
        :param metre_filter: фильтр по метру в текущем состоянии.
        :param rhyme_filter: фильтр по рифме в текущем состоянии.
        :param probability: вероятность пути.
        :param line_ends: окончания строк (в номерах слов).
        """
        self.indices = indices  # type: List[int]
        self.metre_filter = metre_filter  # type: MetreFilter
        self.rhyme_filter = rhyme_filter  # type: RhymeFilter
        self.probability = probability  # type: float
        self.line_ends = line_ends  # type: List[int]

    def put_line_end(self):
        """
        Пометить текущее слово как конец стоки.
        """
        self.line_ends.append(len(self.indices))

    def get_words(self, vocabulary: Vocabulary) -> List[str]:
        """
        Получить слова текущего пути.
        
        :param vocabulary: словарь.
        :return: слова.
        """
        return [vocabulary.get_word(word_index).text.lower() for word_index in self.indices]

    def get_poem(self, vocabulary: Vocabulary) -> str:
        """
        Получить стихотворение этого пути.
        
        :param vocabulary: словарь.
        :return: стихотворение.
        """
        words = self.get_words(vocabulary)
        prev_end = 0
        lines = []
        for end in self.line_ends:
            line = " ".join(list(reversed(words[prev_end:end]))).capitalize()
            prev_end = end
            lines.append(line)
        return "\n".join(list(reversed(lines))) + "\n"

    def get_current_model(self, model_container: ModelContainer, vocabulary: Vocabulary, use_rhyme: bool=False) -> np.array:
        """
        Получить фильтрованные вероятности следующего слова.
        
        :param model_container: контейнер модели.
        :param vocabulary: словарь.
        :param use_rhyme: использовать ли фильтр по рифме?
        :return: модель (вероятности следующего слова).
        """
        model = model_container.get_model(self.indices)
        model = self.metre_filter.filter_model(model, vocabulary)
        if use_rhyme:
            model = self.rhyme_filter.filter_model(model, vocabulary)
        return model

    def is_empty(self) -> bool:
        return len(self.indices) == 0

    def __str__(self):
        return str(self.metre_filter.position) + " " + str(self.rhyme_filter.position) + " " + \
               str(self.probability) + " " + str(self.indices) + " " + str(self.line_ends)

    def __repr__(self):
        return self.__str__()


class Generator(object):
    """
    Генератор стихов
    """
    def __init__(self, model_container: ModelContainer, vocabulary: Vocabulary,
                 word_form_vocabulary: WordFormVocabulary=None):
        """
        :param model_container: модель с методом get_model.
        :param vocabulary: словарь с индексами.
        """
        self.model_container = model_container  # type: ModelContainer
        self.vocabulary = vocabulary  # type: Vocabulary
        self.word_form_vocabulary = word_form_vocabulary  # type: WordFormVocabulary

    def generate_poem(self, metre_schema: str="+-", rhyme_pattern: str="aabb", n_syllables: int=8,
                      letters_to_rhymes: dict=None, beam_width: int=4) -> str:
        """
        Генерация стихотворения с выбранными параметрами.

        :param metre_schema: схема метра.
        :param rhyme_pattern: схема рифмы.
        :param n_syllables: количество слогов в строке.
        :param letters_to_rhymes: заданные рифмы.
        :param beam_width: ширина лучевого поиска.
        :return: стихотворение.
        """
        metre_pattern = ""
        while len(metre_pattern) <= n_syllables:
            metre_pattern += metre_schema
        metre_pattern = metre_pattern[:n_syllables]
        metre_filter = MetreFilter(metre_pattern)
        rhyme_filter = RhymeFilter(rhyme_pattern, letters_to_rhymes, self.word_form_vocabulary, score_border=5)

        result_paths = []
        empty_path = BeamPath([], metre_filter, rhyme_filter, 1.0, [])
        paths = [empty_path]
        while len(paths) != 0:
            paths = self.__top_paths(paths, beam_width)
            for path in copy.deepcopy(paths):
                paths.pop(0)
                paths += self.generate_line_beam(path, beam_width)
            paths, to_result = self.__filter_path_by_rhyme(paths)
            result_paths += to_result
        if len(result_paths) == 0:
            return None
        best_path = self.__top_paths(result_paths, 1)[0]
        return best_path.get_poem(self.vocabulary)

    def generate_line_beam(self, path, beam_width=5):
        """
        Генерация строки (новых путей) с учётом текущего пути.
        
        :param path: текущий путь.
        :param beam_width: ширина луча (количество путей)
        :return: новые пути.
        """
        path.metre_filter.reset()
        paths = self.generate_paths(path, use_rhyme=True)
        result_paths = []
        while len(paths) != 0:
            paths = self.__top_paths(paths, beam_width)
            for i, path in enumerate(copy.copy(paths)):
                new_paths = self.generate_paths(path, beam_width=beam_width, use_rhyme=False)
                paths.pop(0)
                paths += new_paths
            paths, to_result = self.__filter_path_by_metre(paths)
            result_paths += to_result
        result_paths = self.__top_paths(result_paths, 10)
        for i in range(len(result_paths)):
            result_paths[i].put_line_end()
        return result_paths

    def generate_paths(self, path: BeamPath, beam_width: int=10, use_rhyme: bool=False):
        """
        Расширение 1 пути на beam_width путей новыми словами.
        
        :param path: оригинальный путь.
        :param beam_width: колцичество новых путей.
        :param use_rhyme: использовать ли фильтр по рифме.
        :param use_top: брать ли топ по языковой модели или делать случайный выбор.
        :return: новые пути.
        """
        model = path.get_current_model(self.model_container, self.vocabulary, use_rhyme)
        if np.sum(model) == 0.0:
            return []
        if len(path.indices) != 0:
            new_indices = Generator.__choose(model, beam_width)
        else:
            new_indices = Generator.__choose_uniform(self.vocabulary.size(), beam_width)
        new_paths = []
        for index in new_indices:
            word = self.vocabulary.get_word(index)
            word_probability = model[index]
            metre_filter = copy.copy(path.metre_filter)
            metre_filter.pass_word(word)
            rhyme_filter = copy.copy(path.rhyme_filter)
            if use_rhyme:
                rhyme_filter.letters_to_rhymes = copy.deepcopy(path.rhyme_filter.letters_to_rhymes)
                rhyme_filter.pass_word(word)
            new_paths.append(BeamPath(path.indices+[index], metre_filter, rhyme_filter,
                                      path.probability * word_probability, copy.copy(path.line_ends)))
        return new_paths

    def generate_poem_by_line(self, line: str, rhyme_pattern: str, stress_predictor: StressPredictor) -> str:
        """
        Генерация стихотвторения по одной строчке.

        :param stress_predictor: классификатор.
        :param line: строчка.
        :param rhyme_pattern: шаблон рифмы.
        :return: стихотворение
        """
        markup, result = MetreClassifier.improve_markup(Markup.process_text(line, stress_predictor))
        rhyme_word = markup.lines[0].words[-1]
        count_syllables = sum([len(word.syllables) for word in markup.lines[0].words])
        metre_pattern = CompilationsSingleton.get().get_patterns(result.metre, count_syllables)[0]
        metre_pattern = metre_pattern.lower().replace("s", "+").replace("u", "-")
        letters_to_rhymes = {rhyme_pattern[0]: {rhyme_word}}
        generated = self.generate_poem(metre_pattern, rhyme_pattern, len(metre_pattern), letters_to_rhymes)
        poem = line + "\n" + "\n".join(generated.split("\n")[1:])
        return poem

    @staticmethod
    def __top_paths(paths, n):
        if len(paths) <= n:
            return paths
        max_indices = np.array([p.probability for p in paths]).argsort()[-n:][::-1]
        max_paths = [path for i, path in enumerate(paths) if i in max_indices]
        return max_paths

    @staticmethod
    def __filter_path_by_metre(paths):
        result_paths = [path for path in paths if path.metre_filter.position == -1]
        ok_paths = [path for path in paths if path.metre_filter.position > -1]
        return ok_paths, result_paths

    @staticmethod
    def __filter_path_by_rhyme(paths):
        result_paths = [path for path in paths if path.rhyme_filter.position == -1]
        ok_paths = [path for path in paths if path.rhyme_filter.position > -1]
        return ok_paths, result_paths

    @staticmethod
    def __choose(model: np.array, n: int=1):
        """
        Выбор слова из языковой модели.

        :param model: языковая модель.
        :return: слово из модели.
        """
        norm_model = model / np.sum(model)
        return choice(range(len(norm_model)), n, p=norm_model)

    @staticmethod
    def __choose_uniform(size: int, n: int=1):
        return [np.random.randint(1, size) for _ in range(n)]

    @staticmethod
    def __top(model: np.array, n: int=1):
        return [i for i in model.argsort()[-n:][::-1] if model[i] != 0.0]