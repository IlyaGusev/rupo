# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Набор внешних методов для работы с библиотекой.

import os
from typing import List, Tuple, Dict

from rulm.language_model import LanguageModel

from rupo.files.reader import FileType, Reader
from rupo.files.writer import Writer
from rupo.g2p.graphemes import Graphemes
from rupo.g2p.rnn import RNNG2PModel
from rupo.main.markup import Markup
from rupo.metre.metre_classifier import MetreClassifier, ClassificationResult
from rupo.rhymes.rhymes import Rhymes
from rupo.settings import RU_G2P_DEFAULT_MODEL, EN_G2P_DEFAULT_MODEL, ZALYZNYAK_DICT, CMU_DICT, DATA_DIR, DICT_DIR
from rupo.stress.predictor import StressPredictor, CombinedStressPredictor
from rupo.main.vocabulary import StressVocabulary, inflate_stress_vocabulary
from rupo.generate.generator import Generator

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.common.util import END_SYMBOL
from rulm.transform import ExcludeTransform


class Engine:
    def __init__(self, language="ru"):
        self.language = language  # type: str
        self.vocabulary = None  # type: StressVocabulary
        self.generator = None  # type: Generator
        self.g2p_models = dict()  # type: Dict[str, RNNG2PModel]
        self.stress_predictors = dict()  # type: Dict[str, StressPredictor]

    def load(self, stress_model_path: str, zalyzniak_dict: str, raw_stress_dict_path=None,
             stress_trie_path=None):
        self.g2p_models = dict()
        self.stress_predictors = dict()
        if not os.path.isdir(DATA_DIR):
            os.makedirs(DATA_DIR)
        if not os.path.isdir(DICT_DIR):
            os.makedirs(DICT_DIR)
        self.get_stress_predictor(self.language, stress_model_path, raw_stress_dict_path,
                                  stress_trie_path, zalyzniak_dict)

    def get_vocabulary(self, dump_path: str, markup_path: str) -> StressVocabulary:
        if self.vocabulary is None:
            self.vocabulary = StressVocabulary()
            if os.path.isfile(dump_path):
                self.vocabulary.load(dump_path)
            elif markup_path is not None:
                self.vocabulary.parse(markup_path)
        return self.vocabulary

    def get_generator(self,
                      model_path: str,
                      token_vocab_path: str,
                      stress_vocab_dump_path: str) -> Generator:
        if self.generator is None:
            assert os.path.isdir(model_path) and os.path.isdir(token_vocab_path)
            vocabulary = Vocabulary.from_files(token_vocab_path)
            stress_vocabulary = StressVocabulary()
            if not os.path.isfile(stress_vocab_dump_path):
                stress_vocabulary = inflate_stress_vocabulary(vocabulary, self.get_stress_predictor())
                stress_vocabulary.save(stress_vocab_dump_path)
            else:
                stress_vocabulary.load(stress_vocab_dump_path)

            eos_index = vocabulary.get_token_index(END_SYMBOL)
            unk_index = vocabulary.get_token_index(DEFAULT_OOV_TOKEN)
            exclude_transform = ExcludeTransform((unk_index, eos_index))

            model = LanguageModel.load(model_path, vocabulary_dir=token_vocab_path,
                                       transforms=[exclude_transform, ])
            self.generator = Generator(model, vocabulary, stress_vocabulary, eos_index)
        return self.generator

    def get_stress_predictor(self, language="ru", stress_model_path: str=None, raw_stress_dict_path=None,
                             stress_trie_path=None, zalyzniak_dict=ZALYZNYAK_DICT, cmu_dict=CMU_DICT):
        if self.stress_predictors.get(language) is None:
            self.stress_predictors[language] = CombinedStressPredictor(language, stress_model_path,
                                                                       raw_stress_dict_path, stress_trie_path,
                                                                       zalyzniak_dict, cmu_dict)
        return self.stress_predictors[language]

    def get_g2p_model(self, language="ru", model_path=None):
        if self.g2p_models.get(language) is None:
            self.g2p_models[language] = RNNG2PModel(language=language)
            if language == "ru" and model_path is None:
                model_path = RU_G2P_DEFAULT_MODEL
            elif language == "en" and model_path is None:
                model_path = EN_G2P_DEFAULT_MODEL
            else:
                return None
            self.g2p_models[language].load(model_path)
        return self.g2p_models[language]

    def get_stresses(self, word: str, language: str="ru") -> List[int]:
        """
        :param word: слово.
        :param language: язык.
        :return: ударения слова.
        """
        return self.get_stress_predictor(language).predict(word)

    @staticmethod
    def get_word_syllables(word: str) -> List[str]:
        """
        :param word: слово.
        :return: его слоги.
        """
        return [syllable.text for syllable in Graphemes.get_syllables(word)]

    @staticmethod
    def count_syllables(word: str) -> int:
        """
        :param word: слово.
        :return: количество слогов в нём.
        """
        return len(Graphemes.get_syllables(word))

    def get_markup(self, text: str, language: str="ru") -> Markup:
        """
        :param text: текст.
        :param language: язык.
        :return: его разметка по словарю.
        """
        return Markup.process_text(text, self.get_stress_predictor(language))

    def get_improved_markup(self, text: str, language: str="ru") -> Tuple[Markup, ClassificationResult]:
        """
        :param text: текст.
        :param language: язык.
        :return: его разметка по словарю, классификатору метру и  ML классификатору.
        """
        markup = Markup.process_text(text, self.get_stress_predictor(language))
        return MetreClassifier.improve_markup(markup)

    def classify_metre(self, text: str, language: str="ru") -> str:
        """
        :param text: текст.
        :param language: язык.
        :return: его метр.
        """
        return MetreClassifier.classify_metre(Markup.process_text(text, self.get_stress_predictor(language))).metre

    def generate_markups(self, input_path: str, input_type: FileType, output_path: str, output_type: FileType) -> None:
        """
        Генерация разметок по текстам.

        :param input_path: путь к папке/файлу с текстом.
        :param input_type: тип файлов с текстов.
        :param output_path: путь к файлу с итоговыми разметками.
        :param output_type: тип итогового файла.
        """
        markups = Reader.read_markups(input_path, input_type, False, self.get_stress_predictor())
        writer = Writer(output_type, output_path)
        writer.open()
        for markup in markups:
            writer.write_markup(markup)
        writer.close()

    def is_rhyme(self, word1: str, word2: str) -> bool:
        """
        :param word1: первое слово.
        :param word2: второе слово.
        :return: рифмуются ли слова.
        """
        markup_word1 = self.get_markup(word1).lines[0].words[0]
        markup_word1.set_stresses(self.get_stresses(word1))
        markup_word2 = self.get_markup(word2).lines[0].words[0]
        markup_word2.set_stresses(self.get_stresses(word2))
        return Rhymes.is_rhyme(markup_word1, markup_word2)

    def generate_poem(self,
                      model_path: str,
                      token_vocab_path: str=None,
                      stress_vocab_path: str=None,
                      metre_schema: str="-+",
                      rhyme_pattern: str="abab",
                      n_syllables: int=8,
                      sampling_k: int=None,
                      beam_width: int=None,
                      seed: int=1337,
                      temperature: float=1.0,
                      last_text: str="") -> str:
        """
        Сгенерировать стих. Нужно задать либо sampling_k, либо beam_width.

        :param model_path: путь к модели.
        :param token_vocab_path: путь к словарю.
        :param stress_vocab_path: путь к словарю ударений.
        :param metre_schema: схема метра.
        :param rhyme_pattern: схема рифм.
        :param n_syllables: количество слогов в строке.
        :param sampling_k: top-k при семплинге
        :param beam_width: ширина лучевого поиска.
        :param seed: seed
        :param temperature: температура генерации
        :param last_text: последняя строчка
        :return: стих. None, если генерация не была успешной.
        """
        token_vocab_path = token_vocab_path or os.path.join(model_path, "vocabulary")
        stress_vocab_path = stress_vocab_path or os.path.join(model_path, "stress.pickle")
        generator = self.get_generator(model_path, token_vocab_path, stress_vocab_path)
        poem = generator.generate_poem(
            metre_schema=metre_schema,
            rhyme_pattern=rhyme_pattern,
            n_syllables=n_syllables,
            sampling_k=sampling_k,
            beam_width=beam_width,
            temperature=temperature,
            seed=seed,
            last_text=last_text
        )
        return poem

    def get_word_rhymes(self, word: str, vocab_dump_path: str, markup_path: str=None) -> List[str]:
        """
        Поиск рифмы для данного слова.

        :param word: слово.
        :param vocab_dump_path: путь, куда сохраняется словарь.
        :param markup_path: путь к разметкам.
        :return: список рифм.
        """
        markup_word = self.get_markup(word).lines[0].words[0]
        markup_word.set_stresses(self.get_stresses(word))
        rhymes = []
        vocabulary = self.get_vocabulary(vocab_dump_path, markup_path)
        for i in range(vocabulary.size()):
            if Rhymes.is_rhyme(markup_word, vocabulary.get_word(i)):
                rhymes.append(vocabulary.get_word(i).text.lower())
        return rhymes
