# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Набор внешних методов для работы с библиотекой.

from typing import List, Tuple

from rupo.main.phonetics import Phonetics
from rupo.main.markup import Markup
from rupo.accents.dict import AccentDict
from rupo.accents.classifier import MLAccentClassifier
from rupo.metre.metre_classifier import MetreClassifier, ClassificationResult
from rupo.files.reader import FileTypeEnum, Reader
from rupo.files.writer import Writer
from rupo.rhymes.rhymes import Rhymes
from rupo.generate.markov import MarkovModelContainer
from rupo.generate.generator import Generator
from rupo.util.vocabulary import Vocabulary


class Global:
    """
    Глобальные ресурсы.
    """
    accent_dict = None
    accent_classifier = None
    vocabulary = None
    markov = None
    generator = None

    @classmethod
    def get_dict(cls):
        if cls.accent_dict is None:
            cls.accent_dict = AccentDict()
        return cls.accent_dict

    @classmethod
    def get_classifier(cls):
        if cls.accent_classifier is None:
            cls.accent_classifier = MLAccentClassifier(cls.get_dict())
        return cls.accent_classifier

    @classmethod
    def get_vocabulary(cls, dump_path, markup_path):
        if cls.vocabulary is None:
            cls.vocabulary = Vocabulary(dump_path, markup_path)
        return cls.vocabulary

    @classmethod
    def get_markov(cls, dump_path, vocab_dump_path, markup_path):
        if cls.markov is None:
            vocab = cls.get_vocabulary(vocab_dump_path, markup_path)
            cls.markov = MarkovModelContainer(dump_path, vocab, markup_path)
        return cls.markov

    @classmethod
    def get_generator(cls, dump_path, vocab_dump_path, markup_path):
        if cls.generator is None:
            cls.generator = Generator(cls.get_markov(dump_path, vocab_dump_path, markup_path),
                                      cls.get_vocabulary(vocab_dump_path, markup_path))
        return cls.generator


def get_accent(word: str) -> int:
    """
    :param word: слово.
    :return: ударение слова.
    """
    return Phonetics.get_improved_word_accent(word, Global.get_dict(), Global.get_classifier())


def get_word_syllables(word: str) -> List[str]:
    """
    :param word: слово.
    :return: его слоги.
    """
    return [syllable.text for syllable in Phonetics.get_word_syllables(word)]


def count_syllables(word: str) -> int:
    """
    :param word: слово.
    :return: количество слогов в нём.
    """
    return len(Phonetics.get_word_syllables(word))


def get_markup(text: str) -> Markup:
    """
    :param text: текст.
    :return: его разметка по словарю.
    """
    return Phonetics.process_text(text, Global.get_dict())


def get_improved_markup(text: str) -> Tuple[Markup, ClassificationResult]:
    """
    :param text: текст.
    :return: его разметка по словарю, классификатору метру и  ML классификатору.
    """
    markup = Phonetics.process_text(text, Global.get_dict())
    return MetreClassifier.improve_markup(markup, Global.get_classifier())


def classify_metre(text: str) -> str:
    """
    :param text: текст.
    :return: его метр.
    """
    return MetreClassifier.classify_metre(Phonetics.process_text(text, Global.get_dict())).metre


def generate_markups(input_path: str, input_type: FileTypeEnum, output_path: str, output_type: FileTypeEnum) -> None:
    """
    Генерация разметок по текстам.

    :param input_path: путь к папке/файлу с текстом.
    :param input_type: тип файлов с текстов.
    :param output_path: путь к файлу с итоговыми разметками.
    :param output_type: тип итогового файла.
    """
    markups = Reader.read_markups(input_path, input_type, False, Global.get_dict(), Global.get_classifier())
    writer = Writer(output_type, output_path)
    writer.open()
    for markup in markups:
        writer.write_markup(markup)
    writer.close()


def is_rhyme(word1: str, word2: str) -> bool:
    """
    :param word1: первое слово.
    :param word2: второе слово.
    :return: рифмуются ли слова.
    """
    markup_word1 = get_markup(word1).lines[0].words[0]
    markup_word1.set_accents([get_accent(word1)])
    markup_word2 = get_markup(word2).lines[0].words[0]
    markup_word2.set_accents([get_accent(word2)])
    return Rhymes.is_rhyme(markup_word1, markup_word2)


def generate_poem(markup_path: str, dump_path: str, vocab_dump_path: str, metre_schema: str="-+",
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
    generator = Global.get_generator(dump_path, vocab_dump_path, markup_path)
    return generator.generate_poem(metre_schema, rhyme_pattern, n_syllables)


def generate_poem_by_line(markup_path: str, dump_path: str, vocab_dump_path: str,
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
    generator = Global.get_generator(dump_path, vocab_dump_path, markup_path)
    return generator.generate_poem_by_line(line, rhyme_pattern, Global.get_dict(), Global.get_classifier())


def get_word_rhymes(word: str, vocab_dump_path: str, markup_path: str=None) -> List[str]:
    """
    Поиск рифмы для данного слова.

    :param word: слово.
    :param vocab_dump_path: путь, куда сохраняется словарь.
    :param markup_path: путь к разметкам.
    :return: список рифм.
    """
    markup_word = get_markup(word).lines[0].words[0]
    markup_word.set_accents([get_accent(word)])
    rhymes = []
    vocabulary = Global.get_vocabulary(vocab_dump_path, markup_path)
    for i in range(len(vocabulary.words)):
        if Rhymes.is_rhyme(markup_word, vocabulary.get_word(i)):
            rhymes.append(vocabulary.get_word(i).text.lower())
    return rhymes
