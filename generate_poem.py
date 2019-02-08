import os
import logging
from typing import List, Tuple, Dict

from rupo.generate.generator import Generator

from rupo.stress.predictor import CombinedStressPredictor
from rupo.main.vocabulary import StressVocabulary
from rupo.generate.transforms import PoemTransform

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.common.util import END_SYMBOL
from rulm.language_model import LanguageModel
from rulm.transform import ExcludeTransform


def inflate_stress_vocabulary(vocabulary: Vocabulary):
    vocab = StressVocabulary()
    stress_predictor = CombinedStressPredictor()
    for index, word in vocabulary.get_index_to_token_vocabulary("tokens").items():
        stresses = [Stress(pos, Stress.Type.PRIMARY) for pos in stress_predictor.predict(word)]
        word = StressedWord(word, set(stresses))
        vocab.add_word(word, index)
    return vocab


def get_generator(model_path: str,
                  token_vocab_path: str,
                  stress_vocab_dump_path: str) -> Generator:
    assert os.path.isdir(model_path) and os.path.isdir(token_vocab_path)
    vocabulary = Vocabulary.from_files(token_vocab_path)
    stress_vocabulary = StressVocabulary()
    if not os.path.isfile(stress_vocab_dump_path):
        stress_vocabulary = inflate_stress_vocabulary(vocabulary)
        stress_vocabulary.save(stress_vocab_dump_path)
    else:
        stress_vocabulary.load(stress_vocab_dump_path)

    eos_index = vocabulary.get_token_index(END_SYMBOL)
    unk_index = vocabulary.get_token_index(DEFAULT_OOV_TOKEN)
    exclude_transform = ExcludeTransform((unk_index, eos_index))

    model = LanguageModel.load(model_path, vocabulary_dir=token_vocab_path,
                               transforms=[exclude_transform, ])
    generator = Generator(model, vocabulary, stress_vocabulary, eos_index)
    return generator

if __name__ == "__main__":
    model_path = "/Users/ilya-gusev/Projects/RuPo/stihi_model"
    vocab_path = os.path.join(model_path, "vocabulary")
    stress_path = os.path.join(model_path, "stress_vocab.pickle")
    metre_schema = "+-"
    rhyme_pattern = "abab"
    n_syllables = 10
    temperature = 1.0
    seed = 1339
    sampling_k = 50000
    beam_width = None

    generator = get_generator(model_path, vocab_path, stress_path)
    for seed in range(54, 100):
        print(seed)
        try:
            poem = generator.generate_poem(
                metre_schema=metre_schema,
                rhyme_pattern=rhyme_pattern,
                n_syllables=n_syllables,
                samling_k=sampling_k,
                beam_width=beam_width,
                temperature=temperature,
                seed=seed,
                last_text="будет классная игрушка"
            )
        except AssertionError as e:
            print(e)
            continue
        print(poem)
