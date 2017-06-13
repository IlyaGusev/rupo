# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты генератора.

# from rupo.generate.lstm import LSTMModelContainer
# from rupo.generate.generator import Generator
# from rupo.settings import GENERATOR_VOCAB_PATH
# from rupo.main.vocabulary import Vocabulary
#
# if __name__ == "__main__":
#     lstm = LSTMModelContainer()
#     vocabulary = Vocabulary(GENERATOR_VOCAB_PATH)
#     lemmatized_vocabulary = lstm.lemmatized_vocabulary
#     generator = Generator(lstm, vocabulary, lemmatized_vocabulary)
#     for i in range(10):
#         print(generator.generate_poem())