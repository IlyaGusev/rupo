# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Набор внешних методов для работы с библиотекой.

from typing import List, Tuple

from rupo.stress.classifier import MLStressClassifier
from rupo.stress.dict import StressDict
from rupo.files.reader import FileType, Reader
from rupo.files.writer import Writer
from rupo.generate.generator import Generator
from rupo.generate.markov import MarkovModelContainer
from rupo.main.markup import Markup
from rupo.main.phonetics import Phonetics
from rupo.main.vocabulary import Vocabulary
from rupo.metre.metre_classifier import MetreClassifier, ClassificationResult
from rupo.rhymes.rhymes import Rhymes


class Engine:
    def __init__(self, language="ru"):
        self.language = language  # type: str
        self.stress_dict = None  # type: StressDict
        self.stress_classifier = None  # type: MLStressClassifier
        self.vocabulary = None  # type: Vocabulary
        self.markov = None  # type: MarkovModelContainer
        self.generator = None  # type: Generator

    def load(self):
        self.stress_dict = None
        self.stress_classifier = None
        self.get_dict()
        self.get_classifier()

    def get_dict(self) -> StressDict:
        if self.stress_dict is None:
            self.stress_dict = StressDict()
        return self.stress_dict

    def get_classifier(self) -> MLStressClassifier:
        if self.stress_classifier is None:
            self.stress_classifier = MLStressClassifier(self.get_dict())
        return self.stress_classifier

    def get_vocabulary(self, dump_path: str, markup_path: str) -> Vocabulary:
        if self.vocabulary is None:
            self.vocabulary = Vocabulary(dump_path, markup_path)
        return self.vocabulary

    def get_markov(self, dump_path: str, vocab_dump_path: str, markup_path: str) -> MarkovModelContainer:
        if self.markov is None:
            vocab = self.get_vocabulary(vocab_dump_path, markup_path)
            self.markov = MarkovModelContainer(dump_path, vocab, markup_path)
        return self.markov

    def get_generator(self, dump_path: str, vocab_dump_path: str, markup_path: str) -> Generator:
        if self.generator is None:
            self.generator = Generator(self.get_markov(dump_path, vocab_dump_path, markup_path),
                                       self.get_vocabulary(vocab_dump_path, markup_path))
        return self.generator

    def get_stress(self, word: str) -> int:
        """
        :param word: слово.
        :return: ударение слова.
        """
        return Phonetics.get_improved_word_stress(word, self.get_dict(), self.get_classifier())

    @staticmethod
    def get_word_syllables(word: str) -> List[str]:
        """
        :param word: слово.
        :return: его слоги.
        """
        return [syllable.text for syllable in Phonetics.get_word_syllables(word)]

    @staticmethod
    def count_syllables(word: str) -> int:
        """
        :param word: слово.
        :return: количество слогов в нём.
        """
        return len(Phonetics.get_word_syllables(word))

    def get_markup(self, text: str) -> Markup:
        """
        :param text: текст.
        :return: его разметка по словарю.
        """
        return Phonetics.process_text(text, self.get_dict())

    def get_improved_markup(self, text: str) -> Tuple[Markup, ClassificationResult]:
        """
        :param text: текст.
        :return: его разметка по словарю, классификатору метру и  ML классификатору.
        """
        markup = Phonetics.process_text(text, self.get_dict())
        return MetreClassifier.improve_markup(markup, self.get_classifier())

    def classify_metre(self, text: str) -> str:
        """
        :param text: текст.
        :return: его метр.
        """
        return MetreClassifier.classify_metre(Phonetics.process_text(text, self.get_dict())).metre

    def generate_markups(self, input_path: str, input_type: FileType, output_path: str, output_type: FileType) -> None:
        """
        Генерация разметок по текстам.

        :param input_path: путь к папке/файлу с текстом.
        :param input_type: тип файлов с текстов.
        :param output_path: путь к файлу с итоговыми разметками.
        :param output_type: тип итогового файла.
        """
        markups = Reader.read_markups(input_path, input_type, False, self.get_dict(), self.get_classifier())
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
        markup_word1.set_stresses([self.get_stress(word1)])
        markup_word2 = self.get_markup(word2).lines[0].words[0]
        markup_word2.set_stresses([self.get_stress(word2)])
        return Rhymes.is_rhyme(markup_word1, markup_word2)

    def generate_poem(self, markup_path: str, dump_path: str, vocab_dump_path: str, metre_schema: str="-+",
                      rhyme_pattern: str="abab", n_syllables: int=8) -> str:
        """
        Сгенерировать стих по данным из разметок.

        :param markup_path: путь к разметкам.
        :param dump_path: путь, куда сохранять модель.
        :param vocab_dump_path: путь, куда сохранять словарь.
        :param metre_schema: схема метра.
        :param rhyme_pattern: схема рифм.
        :param n_syllables: количество слогов в строке.
        :return: стих.
        """
        generator = self.get_generator(dump_path, vocab_dump_path, markup_path)
        return generator.generate_poem(metre_schema, rhyme_pattern, n_syllables)

    def generate_poem_by_line(self, markup_path: str, dump_path: str, vocab_dump_path: str,
                              line, rhyme_pattern="abab") -> str:
        """
        Сгенерировать стих по первой строчке.

        :param markup_path: путь к разметкам.
        :param dump_path: путь, куда сохраняется модель.
        :param vocab_dump_path: путь, куда сохраняется словарь.
        :param line: первая строчка
        :param rhyme_pattern: схема рифм.
        :return: стих.
        """
        generator = self.get_generator(dump_path, vocab_dump_path, markup_path)
        return generator.generate_poem_by_line(line, rhyme_pattern, self.get_dict(), self.get_classifier())

    def get_word_rhymes(self, word: str, vocab_dump_path: str, markup_path: str=None) -> List[str]:
        """
        Поиск рифмы для данного слова.

        :param word: слово.
        :param vocab_dump_path: путь, куда сохраняется словарь.
        :param markup_path: путь к разметкам.
        :return: список рифм.
        """
        markup_word = self.get_markup(word).lines[0].words[0]
        markup_word.set_stresses([self.get_stress(word)])
        rhymes = []
        vocabulary = self.get_vocabulary(vocab_dump_path, markup_path)
        for i in range(vocabulary.size()):
            if Rhymes.is_rhyme(markup_word, vocabulary.get_word(i)):
                rhymes.append(vocabulary.get_word(i).text.lower())
        return rhymes
