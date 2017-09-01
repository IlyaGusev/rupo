import os
from typing import List

from rupo.g2p.rnn import RNNG2PModel
from rupo.settings import RU_G2P_DEFAULT_MODEL, EN_G2P_DEFAULT_MODEL


class G2PPredictor:
    def predict(self, word: str) -> List[int]:
        raise NotImplementedError()


class RNNG2PPredictor:
    def __init__(self, language: str="ru", g2p_model_path: str=None):
        self.language = language
        self.g2p_model_path = g2p_model_path

        if language == "ru":
            self.__init_language_defaults(RU_G2P_DEFAULT_MODEL)
        elif language == "en":
            self.__init_language_defaults(EN_G2P_DEFAULT_MODEL)
        else:
            raise RuntimeError("Wrong language")

        if not os.path.exists(self.g2p_model_path):
            raise RuntimeError("No g2p model available (or wrong path)")

        self.g2p_model = RNNG2PModel(language=language)
        self.g2p_model.load(self.g2p_model_path)

    def __init_language_defaults(self, g2p_model_path):
        if self.g2p_model_path is None:
            self.g2p_model_path = g2p_model_path

    def predict(self, word: str) -> str:
        word = word.lower()
        return self.g2p_model.predict([word])[0]
