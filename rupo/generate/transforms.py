from copy import copy, deepcopy
import numpy as np
from collections import defaultdict

from rulm.transform import Transform

from rupo.rhymes.rhymes import Rhymes
from rupo.main.vocabulary import StressVocabulary
from rupo.stress.word import StressedWord


class PoemTransform(Transform):
    """
    Фильтр по шаблону метра.
    """
    def __init__(self,
                 stress_vocabulary: StressVocabulary,
                 metre_pattern: str,
                 rhyme_pattern: str,
                 n_syllables: int,
                 eos_index: int,
                 letters_to_rhymes: dict=None,
                 score_border=4):
        self.stress_vocabulary = stress_vocabulary

        self.n_syllables = n_syllables

        mul = n_syllables // len(metre_pattern)
        if n_syllables % len(metre_pattern) != 0:
            mul += 1

        self.metre_pattern = metre_pattern * mul
        self.stress_position = len(self.metre_pattern) - 1
        self.eos_index = eos_index

        self.rhyme_pattern = rhyme_pattern
        self.rhyme_position = len(self.rhyme_pattern) - 1
        self.score_border = score_border

        self.letters_to_rhymes = defaultdict(set)
        if letters_to_rhymes is not None:
            for letter, words in letters_to_rhymes.items():
                for word in words:
                    self.letters_to_rhymes[letter].add(word)

    def __call__(self, probabilities: np.array) -> np.array:
        # print(self.stress_position, self.rhyme_position, np.sum(probabilities > 0))
        if self.rhyme_position < 0 and self.stress_position == len(self.metre_pattern) - 1:
            probabilities = np.zeros(probabilities.shape, dtype="float")
            probabilities[self.eos_index] = 1.
            return probabilities
        for index in range(probabilities.shape[0]):
            word = self.stress_vocabulary.get_word(index)
            is_good_by_stress = self._filter_word_by_stress(word)
            is_good_by_rhyme = True
            if self.stress_position == len(self.metre_pattern) - 1:
                is_good_by_rhyme = self._filter_word_by_rhyme(word)
            if not is_good_by_stress or not is_good_by_rhyme:
                probabilities[index] = 0.
        # print(np.sum(probabilities > 0))
        return probabilities

    def advance(self, index: int):
        word = self.stress_vocabulary.get_word(index)
        syllables_count = len(word.syllables)

        if self.stress_position == len(self.metre_pattern) - 1:
            letter = self.rhyme_pattern[self.rhyme_position]
            self.letters_to_rhymes[letter].add(word)
            self.rhyme_position -= 1

        self.stress_position -= syllables_count

        if self.stress_position < 0:
            self.stress_position = len(self.metre_pattern) - 1

    def _filter_word_by_stress(self, word: StressedWord) -> bool:
        syllables = word.syllables
        syllables_count = len(syllables)
        if syllables_count == 0:
            return False
        if syllables_count > self.stress_position + 1 or self.stress_position - syllables_count == 0:
            return False
        for i in range(syllables_count):
            syllable = syllables[i]
            syllable_number = self.stress_position - syllables_count + i + 1
            if syllables_count >= 2 and syllable.stress == -1 and self.metre_pattern[syllable_number] == "+":
                for j in range(syllables_count):
                    other_syllable = syllables[j]
                    other_syllable_number = other_syllable.number - syllable.number + syllable_number
                    if i != j and other_syllable.stress != -1 and self.metre_pattern[other_syllable_number] == "-":
                        return False
        return True

    def _filter_word_by_rhyme(self, word: StressedWord) -> bool:
        if len(word.syllables) <= 1:
            return False
        rhyming_words = self.letters_to_rhymes[self.rhyme_pattern[self.rhyme_position]]
        if len(rhyming_words) == 0:
            return True
        first_word = list(rhyming_words)[0]

        is_rhyme = Rhymes.is_rhyme(first_word, word,
                                   score_border=self.score_border,
                                   syllable_number_border=2) and first_word.text != word.text
        return is_rhyme

    def __copy__(self):
        obj = type(self)(self.stress_vocabulary, self.metre_pattern, self.rhyme_pattern, self.n_syllables,
                         self.eos_index, self.letters_to_rhymes, self.score_border)
        obj.stress_position = self.stress_position
        obj.rhyme_position = self.rhyme_position
        obj.letters_to_rhymes = deepcopy(self.letters_to_rhymes)
        return obj
