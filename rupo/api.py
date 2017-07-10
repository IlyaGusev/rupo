# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Набор внешних методов для работы с библиотекой.

from typing import List, Tuple, Dict

from rupo.files.reader import FileType, Reader
from rupo.files.writer import Writer
from rupo.generate.markov import MarkovModelContainer
from rupo.generate.lstm import LSTMModelContainer
from rupo.generate.generator import Generator
from rupo.generate.word_form_vocabulary import WordFormVocabulary
from rupo.main.markup import Markup
from rupo.main.vocabulary import Vocabulary
from rupo.metre.metre_classifier import MetreClassifier, ClassificationResult
from rupo.rhymes.rhymes import Rhymes
from rupo.stress.predictor import StressPredictor, CombinedStressPredictor
from rupo.g2p.rnn import RNNG2PModel
from rupo.g2p.graphemes import Graphemes
from rupo.settings import RU_G2P_DEFAULT_MODEL, EN_G2P_DEFAULT_MODEL


class Engine:
    def __init__(self, language="ru"):
        self.language = language  # type: str
        self.vocabulary = None  # type: Vocabulary
        self.markov = None  # type: MarkovModelContainer
        self.markov_generator = None  # type: Generator
        self.lstm_generator = None  # type: Generator
        self.g2p_models = dict()  # type: Dict[str, RNNG2PModel]
        self.stress_predictors = dict()  # type: Dict[str, StressPredictor]

    def load(self, language="ru"):
        self.g2p_models = dict()
        self.stress_predictors = dict()
        self.get_stress_predictor(language)
        self.get_g2p_model(language)

    def get_vocabulary(self, dump_path: str, markup_path: str) -> Vocabulary:
        if self.vocabulary is None:
            self.vocabulary = Vocabulary(dump_path, markup_path)
        return self.vocabulary

    def get_markov(self, dump_path: str, vocab_dump_path: str, markup_path: str, n_grams: int=2, n_poems: int=None) -> MarkovModelContainer:
        if self.markov is None:
            vocab = self.get_vocabulary(vocab_dump_path, markup_path)
            self.markov = MarkovModelContainer(dump_path, vocab, markup_path, n_grams=n_grams, n_poems=n_poems)
        return self.markov

    def get_markov_generator(self, dump_path: str, vocab_dump_path: str, markup_path: str) -> Generator:
        if self.markov_generator is None:
            self.markov_generator = Generator(self.get_markov(dump_path, vocab_dump_path, markup_path, n_grams=2, n_poems=4000),
                                              self.get_vocabulary(vocab_dump_path, markup_path))
        return self.markov_generator

    def get_lstm_generator(self, model_path: str, word_form_vocab_dump_path: str,
                           stress_vocab_dump_path: str, gram_dump_path: str) -> Generator:
        if self.lstm_generator is None:
            lstm = LSTMModelContainer(model_path, word_form_vocab_dump_path, gram_dump_path)
            word_form_vocabulary = WordFormVocabulary(word_form_vocab_dump_path)
            vocabulary = Vocabulary(stress_vocab_dump_path)
            self.lstm_generator = Generator(lstm, vocabulary, word_form_vocabulary)
        return self.lstm_generator

    def get_stress_predictor(self, language="ru"):
        if self.stress_predictors.get(language) is None:
            self.stress_predictors[language] = CombinedStressPredictor(language)
        return self.stress_predictors[language]

    def get_g2p_model(self, language="ru"):
        if self.g2p_models.get(language) is None:
            self.g2p_models[language] = RNNG2PModel(language=language)
            if language == "ru":
                model_path = RU_G2P_DEFAULT_MODEL
            elif language == "en":
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

    def generate_markov_poem(self, markup_path: str, dump_path: str, vocab_dump_path: str, metre_schema: str="-+",
                             rhyme_pattern: str="abab", n_syllables: int=8, beam_width=5) -> str:
        """
        Сгенерировать стих по данным из разметок.

        :param markup_path: путь к разметкам.
        :param dump_path: путь, куда сохранять модель.
        :param vocab_dump_path: путь, куда сохранять словарь.
        :param metre_schema: схема метра.
        :param rhyme_pattern: схема рифм.
        :param n_syllables: количество слогов в строке.
        :param beam_width: ширина лучевого поиска.
        :return: стих.
        """
        generator = self.get_markov_generator(dump_path, vocab_dump_path, markup_path)
        return generator.generate_poem(metre_schema, rhyme_pattern, n_syllables, beam_width=beam_width)

    def generate_poem(self, model_path: str, word_form_vocab_dump_path: str, gram_dump_path: str,
                      stress_vocab_dump_path: str, metre_schema: str="-+",
                      rhyme_pattern: str="abab", n_syllables: int=8, beam_width: int=5) -> str:
        """
        Сгенерировать стих.

        :param model_path: путь к модели.
        :param word_form_vocab_dump_path: путь к дампу словаря словоформ.
        :param gram_dump_path: путь к векторам грамматических значений.
        :param stress_vocab_dump_path: путь к словарю ударений.
        :param metre_schema: схема метра.
        :param rhyme_pattern: схема рифм.
        :param n_syllables: количество слогов в строке.
        :param beam_width: ширина лучевого поиска.
        :return: стих. None, если генерация не была успешной.
        """
        generator = self.get_lstm_generator(model_path, word_form_vocab_dump_path,
                                            stress_vocab_dump_path, gram_dump_path)
        return generator.generate_poem(metre_schema, rhyme_pattern, n_syllables, beam_width=beam_width)

    def generate_poem_by_line(self, model_path: str, word_form_vocab_dump_path: str,
                              stress_vocab_dump_path: str, line: str, rhyme_pattern: str="abab") -> str:
        """
        Сгенерировать стих по первой строчке.

        :param model_path: путь к модели.
        :param word_form_vocab_dump_path: путь к дампу словаря словоформ.
        :param stress_vocab_dump_path: путь к словарю ударений.
        :param line: первая строчка
        :param rhyme_pattern: схема рифм.
        :return: стих. None, если генерация не была успешной.
        """
        generator = self.get_lstm_generator(model_path, word_form_vocab_dump_path, stress_vocab_dump_path)
        return generator.generate_poem_by_line(line, rhyme_pattern, self.get_stress_predictor())

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
