# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для рекуррентной сети g2p.
#
from keras.layers import LSTM

from rupo.g2p.rnn import RNNPhonemePredictor
from rupo.settings import EN_PHONEME_STRESS_PATH, ACCENT_CURRENT_MODEL_DIR, EN_G2P_DICT_PATH, \
    G2P_CURRENT_MODEL_DIR, RU_PHONEME_STRESS_PATH, RU_G2P_DICT_PATH
from rupo.stress.rnn import RNNStressPredictor


def g2p_ru():
    clf = RNNPhonemePredictor(RU_G2P_DICT_PATH, 40, language="ru", rnn=LSTM, units1=256, dropout=0.4)
    clf.build()
    clf.train(G2P_CURRENT_MODEL_DIR, enable_checkpoints=True, checkpoint="/home/yallen/Документы/Python/rupo/rupo/data/g2p_models/12-0.04.hdf5")


def stress_ru():
    clf = RNNStressPredictor(RU_PHONEME_STRESS_PATH, 19, language="ru", rnn=LSTM)
    clf.build()
    clf.train(ACCENT_CURRENT_MODEL_DIR, enable_checkpoints=True)


def g2p_en():
    clf = RNNPhonemePredictor(EN_G2P_DICT_PATH, 40, language="en", rnn=LSTM, units1=256, dropout=0.5)
    clf.build()
    clf.train(G2P_CURRENT_MODEL_DIR, enable_checkpoints=True)


def stress_en():
    clf = RNNStressPredictor(EN_PHONEME_STRESS_PATH, 25, language="en", rnn=LSTM)
    clf.build()
    clf.train(ACCENT_CURRENT_MODEL_DIR, enable_checkpoints=True)

# stress_ru()
# stress_en()
# g2p_en()
g2p_ru()