# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для рекуррентной сети g2p.
#
import os
from keras.layers import GRU, LSTM
from rupo.settings import EN_PHONEME_STRESS_PATH, ACCENT_CURRENT_MODEL_DIR, EN_G2P_DICT_PATH, \
    G2P_CURRENT_MODEL_DIR, RU_PHONEME_STRESS_PATH, RU_G2P_DICT_PATH
from rupo.g2p.rnn_stress import RNNStressPredictor
from rupo.g2p.rnn_g2p import RNNPhonemePredictor


def g2p_ru():
    clf = RNNPhonemePredictor(RU_G2P_DICT_PATH, 35, language="ru", rnn=LSTM, units1=512, dropout=0.1)
    clf.build()
    clf.train(G2P_CURRENT_MODEL_DIR, enable_checkpoints=True)


def stress_ru():
    clf = RNNStressPredictor(RU_PHONEME_STRESS_PATH, 19, language="ru", rnn=LSTM)
    clf.build()
    clf.train(ACCENT_CURRENT_MODEL_DIR, enable_checkpoints=True)


def g2p_en():
    clf = RNNPhonemePredictor(EN_G2P_DICT_PATH, 40, language="en", rnn=LSTM, units1=512, dropout=0.1)
    clf.build()
    clf.train(G2P_CURRENT_MODEL_DIR, enable_checkpoints=True)


def stress_en():
    clf = RNNStressPredictor(EN_PHONEME_STRESS_PATH, 25, language="en", rnn=LSTM)
    clf.build()
    clf.train(ACCENT_CURRENT_MODEL_DIR, enable_checkpoints=True)

# stress_ru()
# stress_en()
# g2p_en()
# g2p_ru()