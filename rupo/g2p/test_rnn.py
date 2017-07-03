# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для рекуррентной сети g2p.
#
from keras.layers import LSTM

from rupo.g2p.rnn import RNNG2PModel
from rupo.settings import EN_PHONEME_STRESS_PATH, ACCENT_CURRENT_MODEL_DIR, EN_G2P_DICT_PATH, \
    G2P_CURRENT_MODEL_DIR, RU_PHONEME_STRESS_PATH, RU_G2P_DICT_PATH
from rupo.stress.rnn import RNNStressModel


def g2p_ru():
    clf = RNNG2PModel(RU_G2P_DICT_PATH, 30, language="ru", rnn=LSTM, units1=512, units2=512, dropout=0.4,
                      batch_size=128, emb_dimension=50)
    clf.build()
    clf.train(G2P_CURRENT_MODEL_DIR, enable_checkpoints=True)


def stress_ru():
    clf = RNNStressModel(RU_PHONEME_STRESS_PATH, 19, language="ru", rnn=LSTM)
    clf.build()
    clf.train(ACCENT_CURRENT_MODEL_DIR, enable_checkpoints=True)


def g2p_en():
    clf = RNNG2PModel(EN_G2P_DICT_PATH, 40, language="en", rnn=LSTM, units1=256, dropout=0.5)
    clf.build()
    clf.train(G2P_CURRENT_MODEL_DIR, enable_checkpoints=True)


def stress_en():
    clf = RNNStressModel(EN_PHONEME_STRESS_PATH, 25, language="en", rnn=LSTM)
    clf.build()
    clf.train(ACCENT_CURRENT_MODEL_DIR, enable_checkpoints=True)

# if __name__ == "__main__":
    # stress_ru()
    # stress_en()
    # g2p_en()
    # g2p_ru()