import argparse

from rupo.api import Engine
from rupo.settings import RU_STRESS_DEFAULT_MODEL, ZALYZNYAK_DICT, GENERATOR_MODEL_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=GENERATOR_MODEL_DIR)
    parser.add_argument('--token-vocab-path', type=str, default=None)
    parser.add_argument('--stress-vocab-path', type=str, default=None)
    parser.add_argument('--metre-schema', type=str, default='+-')
    parser.add_argument('--rhyme-pattern', type=str, default='abab')
    parser.add_argument('--n-syllables', type=int, default=8)
    parser.add_argument('--sampling-k', type=int, default=50000)
    parser.add_argument('--beam-width', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--last-text', type=str, default="")
    parser.add_argument('--count', type=int, default=100)
    args = parser.parse_args()

    kwargs = vars(args)
    count = kwargs.pop('count')

    engine = Engine()
    engine.load(RU_STRESS_DEFAULT_MODEL, ZALYZNYAK_DICT)
    for seed in range(count):
        print(seed)
        try:
            poem = engine.generate_poem(seed=seed, **kwargs)
            print(poem)
        except AssertionError as e:
            print("Error: ", e)
