# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль создания стихотворений.

import copy
from typing import List, Optional

import numpy as np
from numpy.random import choice

from allennlp.data.vocabulary import Vocabulary
from rulm.language_model import LanguageModel

from rupo.generate.filters import MetreFilter, RhymeFilter
from rupo.main.vocabulary import StressVocabulary
from rupo.stress.word import StressedWord, Stress
from rupo.stress.predictor import CombinedStressPredictor


def inflate_stress_vocabulary(vocabulary: Vocabulary):
    vocab = StressVocabulary()
    stress_predictor = CombinedStressPredictor()
    for index, word in vocabulary.get_index_to_token_vocabulary("tokens").items():
        stresses = [Stress(pos, Stress.Type.PRIMARY) for pos in stress_predictor.predict(word)]
        word = StressedWord(word, set(stresses))
        vocab.add_word(word, index)
    return vocab


class BeamPath(object):
    """
    Путь в лучевом поиске.
    """
    def __init__(self,
                 indices: List[int],
                 metre_filter: MetreFilter,
                 rhyme_filter: RhymeFilter,
                 probability: float,
                 line_ends: List[int]):
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

    def get_words(self, vocabulary: StressVocabulary) -> List[str]:
        """
        Получить слова текущего пути.
        
        :param vocabulary: словарь.
        :return: слова.
        """
        return [vocabulary.get_word(word_index).text.lower() for word_index in self.indices]

    def get_poem(self, vocabulary: StressVocabulary) -> str:
        """
        Получить стихотворение этого пути.
        
        :param vocabulary: словарь.
        :return: стихотворение.
        """
        words = self.get_words(vocabulary)
        prev_end = 1
        lines = []
        for end in self.line_ends:
            line = " ".join(list(reversed(words[prev_end:end]))).capitalize()
            prev_end = end
            lines.append(line)
        return "\n".join(list(reversed(lines))) + "\n"

    def get_current_model(self,
                          model: LanguageModel,
                          vocabulary: StressVocabulary,
                          use_rhyme: bool=False) -> np.array:
        """
        Получить фильтрованные вероятности следующего слова.
        
        :param model: контейнер модели.
        :param vocabulary: словарь.
        :param use_rhyme: использовать ли фильтр по рифме?
        :return: модель (вероятности следующего слова).
        """
        model = model.predict_text(self.indices)
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
    def __init__(self,
                 model: LanguageModel,
                 token_vocabulary: Vocabulary,
                 stress_vocabulary: StressVocabulary):
        self.model = model  # type: LanguageModel
        self.token_vocabulary = token_vocabulary  # type: Vocabulary
        self.stress_vocabulary = stress_vocabulary  # type: StressVocabulary

    def generate_poem(self,
                      metre_schema: str="+-",
                      rhyme_pattern: str="aabb",
                      n_syllables: int=8,
                      letters_to_rhymes: dict=None,
                      beam_width: int=4,
                      rhyme_score_border: int=4) -> Optional[str]:
        poem = self.model.beam_decoding("", beam_width=beam_width)
        poem = " ".join(poem.split(" ")[::-1])
        return poem

    def generate_line_beam(self, path, beam_width=5):
        """
        Генерация строки (новых путей) с учётом текущего пути.
        
        :param path: текущий путь.
        :param beam_width: ширина луча (количество путей)
        :return: новые пути.
        """
        path.metre_filter.reset()
        paths = self.generate_paths(path, beam_width, use_rhyme=True)
        result_paths = []
        while len(paths) != 0:
            paths = self.__top_paths(paths, beam_width)
            for i, path in enumerate(copy.copy(paths)):
                new_paths = self.generate_paths(path, beam_width, use_rhyme=False)
                paths.pop(0)
                paths += new_paths
            paths, to_result = self.__filter_path_by_metre(paths)
            result_paths += to_result
        result_paths = self.__top_paths(result_paths, beam_width)
        for i in range(len(result_paths)):
            result_paths[i].put_line_end()
        return result_paths

    def generate_paths(self, path: BeamPath, beam_width: int=10, use_rhyme: bool=False):
        """
        Расширение 1 пути на beam_width путей новыми словами.
        
        :param path: оригинальный путь.
        :param beam_width: колцичество новых путей.
        :param use_rhyme: использовать ли фильтр по рифме.
        :return: новые пути.
        """
        model = path.get_current_model(self.model, self.stress_vocabulary, use_rhyme)
        if np.sum(model) == 0.0:
            return []
        new_indices = Generator.__choose(model, beam_width)
        new_paths = []
        for index in new_indices:
            word = self.token_vocabulary.get_token_from_index(index)
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
    def __choose_uniform(size: int, n: int = 1):
        return [np.random.randint(1, size) for _ in range(n)]

    @staticmethod
    def __choose(model: np.array, n: int=1):
        """
        Выбор слова из языковой модели.

        :param model: языковая модель.
        :param: количество слов.
        :return: индексы слов из модели.
        """
        norm_model = model / np.sum(model)
        try:
            return choice(range(len(norm_model)), n, p=norm_model, replace=False)
        except ValueError:
            return choice(range(len(norm_model)), n, p=norm_model, replace=True)

    @staticmethod
    def __top(model: np.array, n: int=1):
        return [i for i in model.argsort()[-n:][::-1] if model[i] != 0.0]