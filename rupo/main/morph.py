# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Разметка по грамматическим значениям

import os
from typing import List, TextIO

from sentence_splitter import SentenceSplitter
from rnnmorph.predictor import RNNMorphPredictor

from rupo.main.tokenizer import Tokenizer, Token


class Morph:
    @staticmethod
    def get_morph_markup(input_filenames: List[str], output_filename: str):
        """
        Разметка по грамматическим значениям

        :param input_filenames: входные текстовые файлы
        :param output_filename: путь к файлу, куда будет сохранена разметка
        """
        if os.path.exists(output_filename):
            os.remove(output_filename)

        sentence_splitter = SentenceSplitter(language='ru')
        morph_predictor = RNNMorphPredictor()

        for filename in input_filenames:
            with open(filename, "r", encoding="utf-8") as r, open(output_filename, "w+", encoding="utf-8") as w:
                for line in r:
                    Morph.__process_line(line, w, sentence_splitter, morph_predictor)

    @staticmethod
    def __process_line(line: str, output_file: TextIO, sentence_splitter: SentenceSplitter,
                       morph_predictor: RNNMorphPredictor):
        sentences = sentence_splitter.split(line)
        for sentence in sentences:
            words = [token.text for token in Tokenizer.tokenize(sentence)
                     if token.text != '' and token.token_type != Token.TokenType.SPACE]
            if not words:
                continue
            forms = morph_predictor.predict_sentence_tags(words)
            for form in forms:
                if form.pos == "PUNCT":
                    continue
                output_file.write("%s\t%s\t%s\t%s\n" % (form.word, form.normal_form, form.pos, form.tag))
            output_file.write("\n")
