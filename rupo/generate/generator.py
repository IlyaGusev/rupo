# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Модуль создания стихотворений.

from typing import Optional

from allennlp.data.vocabulary import Vocabulary
from rulm.language_model import LanguageModel

from rupo.main.vocabulary import StressVocabulary
from rupo.generate.transforms import PoemTransform


class Generator(object):
    """
    Генератор стихов
    """
    def __init__(self,
                 model: LanguageModel,
                 token_vocabulary: Vocabulary,
                 stress_vocabulary: StressVocabulary,
                 eos_index: int):
        self.model = model  # type: LanguageModel
        self.token_vocabulary = token_vocabulary  # type: Vocabulary
        self.stress_vocabulary = stress_vocabulary  # type: StressVocabulary
        self.eos_index = eos_index

    def generate_poem(self,
                      metre_schema: str="+-",
                      rhyme_pattern: str="aabb",
                      n_syllables: int=8,
                      letters_to_rhymes: dict=None,
                      beam_width: int=None,
                      sampling_k: int=None,
                      rhyme_score_border: int=4,
                      temperature: float=1.0,
                      seed: int=1337,
                      last_text: str="") -> Optional[str]:
        assert beam_width or sampling_k, "Set sampling_k or beam_width"
        self.model.set_seed(seed)

        poem_transform = PoemTransform(
            stress_vocabulary=self.stress_vocabulary,
            metre_pattern=metre_schema,
            rhyme_pattern=rhyme_pattern,
            n_syllables=n_syllables,
            eos_index=self.eos_index,
            letters_to_rhymes=letters_to_rhymes,
            score_border=rhyme_score_border
        )

        if last_text:
            words = last_text.split(" ")
            last_text = " ".join(words[::-1])
            filled_syllables = 0
            for word in last_text.split():
                index = self.token_vocabulary.get_token_index(word)
                word = self.stress_vocabulary.get_word(index)
                syllables_count = len(word.syllables)
                filled_syllables += syllables_count
            poem_transform.stress_position -= filled_syllables
            poem_transform.rhyme_position -= 1
            last_index = self.token_vocabulary.get_token_index(words[-1])
            last_word = self.stress_vocabulary.get_word(last_index)
            poem_transform.letters_to_rhymes[rhyme_pattern[-1]].add(last_word)

        self.model.transforms.append(poem_transform)

        try:
            if beam_width:
                poem = self.model.beam_decoding(last_text, beam_width=beam_width, temperature=temperature)
            elif sampling_k:
                poem = self.model.sample_decoding(last_text, k=sampling_k, temperature=temperature)
            else:
                assert False
        except Exception as e:
            self.model.transforms.pop()
            raise e

        self.model.transforms.pop()

        words = poem.split(" ")
        words = words[::-1]
        result_words = []
        current_n_syllables = 0
        for word in words:
            result_words.append(word)
            index = self.token_vocabulary.get_token_index(word)
            word = self.stress_vocabulary.get_word(index)
            syllables_count = len(word.syllables)
            current_n_syllables += syllables_count
            if n_syllables == current_n_syllables:
                current_n_syllables = 0
                result_words.append("\n")
        poem = " ".join(result_words)
        poem = "\n".join([line.strip() for line in poem.split("\n")])
        return poem

