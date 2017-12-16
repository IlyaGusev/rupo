from typing import List
import subprocess
import numpy as np
from tqdm import tqdm

from rupo.generate.language_model.model_container import ModelContainer
from rupo.main.vocabulary import StressVocabulary
from rupo.stress.word import StressedWord, Stress
from rupo.stress.predictor import CombinedStressPredictor


class RNNLMModelContainer(ModelContainer):
    """
    Контейнер для языковой модели на основе LSTM.
    """
    def __init__(self, exe_path, model_path, vocabulary_path):
        self.exe_path = exe_path
        self.model_path = model_path
        self.vocabulary = StressVocabulary(vocabulary_path)

    def get_model(self, word_indices: List[int]) -> np.array:
        cmd = [self.exe_path, '-rnnlm', self.model_path, '--generate-samples', "1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
        words = []
        for index in word_indices:
            words.append(self.vocabulary.get_word(index).text)
        inp = " ".join(words) + "\n"
        print(inp)
        input_word = inp.encode('utf-8')
        outs = proc.communicate(input_word)[0]
        proc.kill()
        lines = outs.decode('utf-8').split("\n")[:-1]
        model = np.zeros(self.vocabulary.size())
        for i, line in enumerate(lines):
            _, prob = line.strip().split()
            model[i] = float(prob)
        model[0] = 0.0
        model[self.vocabulary.get_word_index(StressedWord("<UKN>", set()))] = 0.0

        return model

    def generate_vocabulary(self, vocab_path, seed='я'):
        cmd = [self.exe_path, '-rnnlm', self.model_path, '--generate-samples', "1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.PIPE, bufsize=0)
        input_word = (seed+"\n").encode('utf-8')
        outs = proc.communicate(input_word)[0]
        lines = outs.decode('utf-8').split("\n")[:-1]
        lines = [line.split() for line in lines]
        proc.kill()

        vocab = StressVocabulary(vocab_path)
        stress_predictor = CombinedStressPredictor()
        for index, (text, _) in tqdm(enumerate(lines), desc="Accenting words"):
            stresses = [Stress(pos, Stress.Type.PRIMARY) for pos in stress_predictor.predict(text)]
            word = StressedWord(text, set(stresses))
            vocab.add_word(word, index)
        vocab.save()