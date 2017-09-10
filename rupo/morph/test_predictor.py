import unittest
from rupo.morph.predictor import MorphPredictor
from rupo.settings import RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP, RU_MORPH_GRAMMEMES_DICT

# import os
# dir_name = "/media/data/Datasets/Morpho/clean"
# lstm = LSTMMorphoAnalysis(external_batch_size=1000, nn_batch_size=128)
# lstm.prepare([os.path.join(dir_name, filename) for filename in os.listdir(dir_name)], RU_MORPH_WORD_VOCAB_DUMP, RU_MORPH_GRAMMEMES_DICT)
# lstm.build()
# lstm.load(RU_MORPH_DEFAULT_MODEL)
# lstm.train([os.path.join(dir_name, filename) for filename in os.listdir(dir_name)], RU_MORPH_DEFAULT_MODEL)


class TestLSTMMorpho(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = MorphPredictor(RU_MORPH_DEFAULT_MODEL, RU_MORPH_WORD_VOCAB_DUMP, RU_MORPH_GRAMMEMES_DICT)

    @classmethod
    def tearDownClass(cls):
        del cls.predictor

    def test_sentence_analysis(self):
        self.assertEqual(self.predictor.predict(['один', 'жил', 'в', 'пустыне', 'рыбак', 'молодой']),
                         ['NUM#Case=Nom|Gender=Masc',
                          'VERB#Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin|Voice=Act',
                          'ADP#_',
                          'NOUN#Case=Loc|Gender=Fem|Number=Sing',
                          'NOUN#Case=Nom|Gender=Masc|Number=Sing',
                          'ADJ#Case=Gen|Degree=Pos|Gender=Fem|Number=Sing'])