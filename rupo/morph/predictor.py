from typing import List
from rupo.morph.lstm import LSTMMorphoAnalysis


class MorphPredictor:
    def __init__(self, model_filename: str, word_vocab_filename: str, gramm_dict: str):
        self.model = LSTMMorphoAnalysis()
        self.model.prepare(word_vocab_filename, gramm_dict)
        self.model.load(model_filename)

    def predict(self, words: List[str]):
        return self.model.predict(words)