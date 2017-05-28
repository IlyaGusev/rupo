# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты генератора.

from rupo.generate.lstm import LSTM_Container
from rupo.generate.generator import Generator


lstm = LSTM_Container()
generator = Generator(lstm, lstm.lemmatized_vocabulary)
print(generator.generate_poem())