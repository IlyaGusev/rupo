# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль создания стихотворений.

from typing import List
import copy

import numpy as np
from numpy.random import choice

from rupo.stress.stress_classifier import MLStressClassifier
from rupo.stress.dict import StressDict
from rupo.generate.filters import Filter, MetreFilter, RhymeFilter
from rupo.main.phonetics import Phonetics
from rupo.main.vocabulary import Vocabulary
from rupo.metre.metre_classifier import MetreClassifier, CompilationsSingleton


class BeamPath:
    def __init__(self, indices, metre_filter, rhyme_filter, probability, line_ends):
        self.indices = indices
        self.metre_filter = metre_filter
        self.probability = probability
        self.rhyme_filter = rhyme_filter
        self.line_ends = line_ends

    def put_line_end(self):
        self.line_ends.append(len(self.indices))

    def get_words(self, vocabulary):
        return [vocabulary.get_word(word_index).text.lower() for word_index in self.indices]

    def __str__(self):
        return str(self.metre_filter.position) + " " + str(self.rhyme_filter.position) + " " + \
               str(self.probability) + " " + str(self.indices) + " " + str(self.line_ends)

    def __repr__(self):
        return self.__str__()


class Generator(object):
    """
    Генератор стихов
    """
    def __init__(self, model_container, vocabulary: Vocabulary, lemmatized_vocabulary=None):
        """
        :param model_container: модель с методом get_model.
        :param vocabulary: словарь с индексами.
        """
        self.model_container = model_container
        self.vocabulary = vocabulary
        self.current_poem_fails = 0
        self.lemmatized_vocabulary = lemmatized_vocabulary

    def generate_poem(self, metre_schema: str="+-", rhyme_pattern: str="abab", n_syllables: int=9,
                      letters_to_rhymes: dict=None) -> str:
        """
        Генерация стихотворения с выбранными параметрами.

        :param metre_schema: схема метра.
        :param rhyme_pattern: схема рифмы.
        :param n_syllables: количество слогов в строке.
        :param letters_to_rhymes: заданные рифмы.
        :return: стихотворение.
        """
        metre_pattern = ""
        while len(metre_pattern) <= n_syllables:
            metre_pattern += metre_schema
        metre_pattern = metre_pattern[:n_syllables]
        metre_filter = MetreFilter(metre_pattern)

        rhyme_filter = RhymeFilter(rhyme_pattern, letters_to_rhymes, self.lemmatized_vocabulary, score_border=4)

        paths = [BeamPath([], metre_filter, rhyme_filter, 1.0, [])]
        result_paths = []
        n = 5
        while len(paths) != 0:
            paths = self.__top_paths(paths, n)
            for path in copy.deepcopy(paths):
                paths.pop(0)
                paths += self.generate_line_beam(path, n)

            paths, to_result = self.__filter_path_by_rhyme(paths)
            result_paths += to_result
        path = self.__top_paths(result_paths, 1)[0]
        words = list(path.get_words(self.vocabulary))
        prev_end = 0
        lines = []
        for end in path.line_ends:
            line = " ".join(list(reversed(words[prev_end:end]))).capitalize()
            prev_end = end
            lines.append(line)
        return "\n".join(list(reversed(lines))) + "\n"

    def generate_line(self, metre_filter: MetreFilter, rhyme_filter: RhymeFilter,
                      prev_word_indices: List[int]) -> List[str]:
        """
        Генерация одной строки с заданными шаблонами метра и рифмы.

        :param metre_filter: фильтр по метру.
        :param rhyme_filter: фильтр по рифме.
        :param prev_word_indices: индексы предыдущих слов.
        :return: слова строка.
        """
        metre_filter.reset()
        result = []
        word_index = self.generate_word(prev_word_indices, [metre_filter, rhyme_filter])
        prev_word_indices.append(word_index)
        result.append(self.vocabulary.get_word(word_index).text.lower())
        while not metre_filter.is_completed():
            word_index = self.generate_word(prev_word_indices, [metre_filter])
            prev_word_indices.append(word_index)
            result.append(self.vocabulary.get_word(word_index).text.lower())
        return result

    def generate_line_beam(self, path, n=5):
        """
        """
        path.metre_filter.reset()
        paths = self.generate_paths(path, is_first=True)
        result_paths = []
        while len(paths) != 0:
            paths = self.__top_paths(paths, n)
            for i, path in enumerate(copy.copy(paths)):
                new_paths = self.generate_paths(path, n=n, is_first=False)
                paths.pop(0)
                for p in new_paths:
                    paths.append(p)
            assert len(paths) <= n*n
            paths, to_result = self.__filter_path_by_metre(paths)
            result_paths += to_result

        if len(result_paths) == 0:
            return []
        result_paths = self.__top_paths(result_paths, 10)
        for i in range(len(result_paths)):
            result_paths[i].put_line_end()
        print("Result path: ", result_paths[0], result_paths[0].get_words(vocabulary=self.vocabulary))
        return result_paths

    def generate_word(self, prev_word_indices: List[int], filters: List[Filter]) -> int:
        """
        Генерация нового слова на основе предыдущих с учётом фильтров.

        :param prev_word_indices: индексы предыдущих слов.
        :param filters: фильтры модели.
        :return: индекс нового слова.
        """
        model = self.model_container.get_model(prev_word_indices)
        for f in filters:
            model = f.filter_model(model, self.vocabulary)
        if sum(model) == 0:
            print("Failed")
            self.current_poem_fails += 1
            model = self.model_container.get_model([])
            for f in filters:
                model = f.filter_model(model, self.vocabulary)
            if sum(model) == 0:
                model = self.model_container.get_model([])
        word_index = Generator.__choose(model)
        word = self.vocabulary.get_word(word_index)
        for f in filters:
            f.pass_word(word)
        return word_index

    def generate_paths(self, path: BeamPath, n: int=10, is_first: bool=False):
        model = self.model_container.get_model(path.indices)
        model = path.metre_filter.filter_model(model, self.vocabulary)
        if is_first:
            model = path.rhyme_filter.filter_model(model, self.vocabulary)
            if sum(model) == 0.0:
                print("Нет рифмы для пути :(", path)
        if sum(model) == 0.0:
            return []
        if len(path.indices) == 0:
            word_indices = Generator.__choose(model, n)
            word_probs = np.array([1.0 for i in range(n)])
        else:
            word_indices, word_probs = Generator.__choose_with_proba(model, n)
        new_paths = []
        for i, index in enumerate(word_indices):
            word = self.vocabulary.get_word(index)
            metre_filter = copy.copy(path.metre_filter)
            rhyme_filter = copy.copy(path.rhyme_filter)
            rhyme_filter.letters_to_rhymes = copy.deepcopy(path.rhyme_filter.letters_to_rhymes)
            metre_filter.pass_word(word)
            if is_first:
                rhyme_filter.pass_word(word)
            new_paths.append(BeamPath(path.indices+[index], metre_filter, rhyme_filter,
                                      path.probability*word_probs[i], copy.copy(path.line_ends)))
        return new_paths

    def generate_poem_by_line(self, line: str, rhyme_pattern: str,
                              stress_dict: StressDict, stress_classifier: MLStressClassifier) -> str:
        """
        Генерация стихотвторения по одной строчке.

        :param stress_dict: словарь ударений.
        :param stress_classifier: классификатор.
        :param line: строчка.
        :param rhyme_pattern: шаблон рифмы.
        :return: стихотворение
        """
        markup, result = MetreClassifier.improve_markup(Phonetics.process_text(line, stress_dict), stress_classifier)
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
        max_indices = np.array([p.probability for p in paths]).argsort()[-n:][::-1]
        max_paths = []
        for i, path in enumerate(paths):
            if i in max_indices:
                max_paths.append(path)
        return max_paths

    @staticmethod
    def __filter_path_by_metre(paths):
        result_paths = []
        ok_paths = []
        for i, path in enumerate(paths):
            if path.metre_filter.position < -1:
                continue
            elif path.metre_filter.position == -1:
                result_paths.append(path)
            else:
                ok_paths.append(path)
        return ok_paths, result_paths

    @staticmethod
    def __filter_path_by_rhyme(paths):
        result_paths = []
        ok_paths = []
        for i, path in enumerate(paths):
            if path.rhyme_filter.position < -1:
                continue
            elif path.rhyme_filter.position == -1:
                result_paths.append(path)
            else:
                ok_paths.append(path)
        return ok_paths, result_paths

    @staticmethod
    def __choose(model: np.array, n: int=1):
        """
        Выбор слова из языковой модели.

        :param model: языковая модель.
        :return: слово из модели.
        """
        if n == 1:
            return choice(range(len(model)), 1, p=model)[0]
        if np.sum(model) != 1.0:
            norm_model = model / np.sum(model)
        else:
            norm_model = model
        return choice(range(len(norm_model)), n, p=norm_model)

    @staticmethod
    def __top(model: np.array, n: int=1):
        return [i for i in model.argsort()[-n:][::-1] if model[i] != 0.0], [p for p in sorted(model)[-n:][::-1] if p != 0.0]

    @staticmethod
    def __choose_with_proba(model: np.array, n: int=1):
        if np.sum(model) != 1.0:
            norm_model = model / np.sum(model)
        else:
            norm_model = model
        indices = choice(range(len(norm_model)), n, p=norm_model)
        proba = [model[i] for i in indices]
        return indices, proba
