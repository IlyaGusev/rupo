import os
import unittest

from rupo.morph.pymorphy import RussianMorphology
from rupo.settings import EXAMPLES_DIR


class TestMorphology(unittest.TestCase):
    def test_markov(self):
        input_filename = os.path.join(EXAMPLES_DIR, "text.txt")
        output_filename = os.path.join(EXAMPLES_DIR, "morph_markup.txt")
        RussianMorphology.do_markup(input_filename, output_filename)
        with open(output_filename, "r", encoding="utf-8") as f:
            self.assertEqual(f.readlines()[2], "жизни	жизнь	NOUN	Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing\n")