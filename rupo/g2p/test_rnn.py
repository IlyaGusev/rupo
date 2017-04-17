# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты для рекуррентной сети g2p.

from rupo.settings import G2P_EN_DICT_PATH, G2P_MODEL_PATH, ACCENT_EN_DICT_PATH
from rupo.g2p.rnn_g2p import RNNPhonemePredictor
from rupo.g2p.rnn_accent import RNNAccentPredictor

clf = RNNAccentPredictor(ACCENT_EN_DICT_PATH, 30, language="en")
clf.build()
clf.train(G2P_MODEL_PATH)