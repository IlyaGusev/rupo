# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты генератора.

import os
import pickle
from rupo.settings import DATA_DIR
from rupo.generate.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.generate.lstm import LSTMGenerator
from rupo.settings import GENERATOR_LSTM_MODEL_PATH, EXAMPLES_DIR
from rupo.main.tokenizer import Tokenizer, Token


# filename = os.path.join("/media", "data", "PoetryMorph.txt")

# vectorizer = GrammemeVectorizer()
# vectorizer.collect_grammemes(filename)
# vectorizer.collect_possible_vectors(filename)
# print(vectorizer.get_ordered_grammemes())
# print(vectorizer.vectors)
#
# vocab = WordFormVocabulary()
# vocab.load_from_corpus(filename, grammeme_vectorizer=vectorizer)
# print(vocab.word_forms)
#
# lstm = LSTMGenerator(nn_batch_size=256, external_batch_size=10000, softmax_size=50000)
# lstm.prepare([filename, ])
# lstm.build()
# lstm.load(GENERATOR_LSTM_MODEL_PATH)
# lstm.train([filename, ])