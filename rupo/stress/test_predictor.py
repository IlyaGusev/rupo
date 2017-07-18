# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Тесты предсказателя ударений.

import unittest

from rupo.stress.predictor import CombinedStressPredictor
from rupo.settings import RU_STRESS_DEFAULT_MODEL, RU_G2P_DEFAULT_MODEL, ZALYZNYAK_DICT, CMU_DICT, RU_WIKI_DICT, \
    RU_GRAPHEME_STRESS_PATH, RU_GRAPHEME_STRESS_TRIE_PATH, RU_ALIGNER_DEFAULT_PATH, RU_G2P_DICT_PATH


class TestStressPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stress_predictor = CombinedStressPredictor(
            stress_model_path=RU_STRESS_DEFAULT_MODEL,
            g2p_model_path=RU_G2P_DEFAULT_MODEL,
            zalyzniak_dict=ZALYZNYAK_DICT,
            ru_wiki_dict=RU_WIKI_DICT,
            cmu_dict=CMU_DICT,
            raw_stress_dict_path=RU_GRAPHEME_STRESS_PATH,
            stress_trie_path=RU_GRAPHEME_STRESS_TRIE_PATH,
            aligner_dump_path=RU_ALIGNER_DEFAULT_PATH,
            g2p_dict_path=RU_G2P_DICT_PATH
        )

    def test_stress(self):
        checks = {
            'я': [0],
            'в': [],
            'он': [0],
            'майка': [1],
            'соломка': [3],
            'изжить': [3],
            'виться': [1],
            'данный': [1],
            'зорька': [1],
            'банка': [1],
            'оттечь': [3],
            'советского': [3],
            'союза': [2],
            'пора': [3, 1],
            'изжила': [5],
            'меда': [1],
            'автоподъёмник': [8],
        }
        for word, pos in checks.items():
            self.assertEqual(sorted(self.stress_predictor.predict(word)), sorted(pos))


