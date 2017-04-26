# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для рекуррентной сети g2p.
#
import os
from keras.layers import GRU, LSTM
from rupo.settings import EN_PHONEME_STRESS_PATH, ACCENT_CURRENT_MODEL_DIR, EN_G2P_DICT_PATH, RU_WIKI_DICT, G2P_CURRENT_MODEL_DIR, RU_PHONEME_STRESS_PATH
from rupo.g2p.rnn_stress import RNNStressPredictor
from rupo.g2p.rnn_g2p import RNNPhonemePredictor

# clf1 = RNNStressPredictor(RU_PHONEME_STRESS_PATH, 21, language="ru", rnn=LSTM)
# clf1.build()
# clf1.train(ACCENT_CURRENT_MODEL_DIR, enable_checkpoints=True)