from rupo.stress.dict import StressDict
from rupo.g2p.rnn_g2p import RNNPhonemePredictor
from rupo.settings import G2P_RU_DICT_PATH, G2P_DEFAULT_RU_MODEL


class ZalyzniakDict:
    @staticmethod
    def do_g2p(d, model):
        for word, accents in d.get_all():
            if word.strip() != "":
                print(word, model.predict(word))

g2p_predictor = RNNPhonemePredictor(G2P_RU_DICT_PATH)
g2p_predictor.build()
g2p_predictor.load(G2P_DEFAULT_RU_MODEL)
ZalyzniakDict.do_g2p(StressDict(), g2p_predictor)