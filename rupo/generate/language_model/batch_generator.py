from typing import List, Tuple

import numpy as np

from rupo.generate.language_model.lstm import CHAR_SET
from rupo.generate.prepare.grammeme_vectorizer import GrammemeVectorizer
from rupo.generate.prepare.word_form import WordForm
from rupo.generate.prepare.word_form_vocabulary import WordFormVocabulary, SEQ_END_WF
from rupo.util.tqdm_open import tqdm_open


class BatchGenerator:
    """
    Генератор наборов примеров для обучения.
    """
    def __init__(self, filenames: List[str], batch_size: int,
                 embedding_size: int, softmax_size: int, sentence_maxlen: int,
                 word_form_vocabulary: WordFormVocabulary, grammeme_vectorizer: GrammemeVectorizer,
                 max_word_len: int):
        """
        :param filenames: имена файлов с морфоразметкой.
        :param batch_size: размер набора семплов.
        :param softmax_size: размер выхода softmax-слоя (=размер итогового набора вероятностей)
        :param sentence_maxlen: маскимальная длина куска предложения.
        :param word_form_vocabulary: словарь словофрм.
        :param grammeme_vectorizer: векторизатор граммем.
        """
        self.filenames = filenames  # type: List[str]
        self.batch_size = batch_size  # type: int
        self.embedding_size = embedding_size # type: int
        self.softmax_size = softmax_size  # type: int
        self.sentence_maxlen = sentence_maxlen  # type: int
        self.max_word_len = max_word_len  # type: int
        self.word_form_vocabulary = word_form_vocabulary  # type: WordFormVocabulary
        self.grammeme_vectorizer = grammeme_vectorizer  # type: GrammemeVectorizer

    def __generate_seqs(self, sentences: List[List[WordForm]]) -> Tuple[List[List[WordForm]], List[WordForm]]:
        """
        Генерация семплов.
        
        :param sentences: куски предложений из словоформ.
        :return: пары (<семпл из словоформ>, следующая за ним словоформа (ответ)).
        """
        seqs, next_words = [], []
        for sentence in sentences:
            # Разворот для генерации справа налево.
            sentence = sentence[::-1]
            for i in range(1, len(sentence)):
                word_form = sentence[i]
                current_part = sentence[max(0, i-self.sentence_maxlen): i]
                # Если следующая словооформа не из предсказываемых, пропускаем её.
                if self.word_form_vocabulary.get_word_form_index(word_form) >= self.softmax_size:
                    continue
                seqs.append(current_part)
                next_words.append(word_form)
        return seqs, next_words

    def __to_tensor(self, sentences: List[List[WordForm]], next_words: List[WordForm]) -> \
            Tuple[np.array, np.array, np.array, np.array]:
        """
        Перевод семплов из словоформ в индексы словоформ, поиск грамматических векторов по индексу.
        
        :param sentences: семплы из словоформ.
        :param next_words: следующие за последовательностями из sentences слова.
        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        n_samples = len(sentences)

        lemmas = np.zeros((n_samples, self.sentence_maxlen), dtype=np.int)
        grammemes = np.zeros((n_samples, self.sentence_maxlen, self.grammeme_vectorizer.grammemes_count()), dtype=np.int)
        chars = np.zeros((n_samples, self.sentence_maxlen, self.max_word_len), dtype=np.int)
        y = np.zeros(n_samples, dtype=np.int)

        for i in range(n_samples):
            sentence = sentences[i]
            next_word = next_words[i]
            lemmas_vector, grammemes_vector, word_char_vectors = \
                self.get_sample(sentence, self.embedding_size, self.max_word_len,
                                word_form_vocabulary=self.word_form_vocabulary,
                                grammeme_vectorizer=self.grammeme_vectorizer)

            lemmas[i, -len(sentence):] = lemmas_vector
            grammemes[i, -len(sentence):] = grammemes_vector
            chars[i, -len(sentence):] = word_char_vectors
            y[i] = min(self.word_form_vocabulary.word_form_indices[next_word], self.softmax_size)
        return lemmas, grammemes, chars, y

    @staticmethod
    def get_sample(sentence, embedding_size: int, max_word_len: int, word_form_vocabulary: WordFormVocabulary,
                   grammeme_vectorizer: GrammemeVectorizer):
        lemmas_vector = [min(word_form_vocabulary.get_lemma_index(x), embedding_size) for x in sentence]
        grammemes_vector = [grammeme_vectorizer.get_vector_by_index(x.gram_vector_index) for x in sentence]
        word_char_vectors = []
        for word in sentence:
            char_indices = np.zeros(max_word_len)
            word_char_indices = [CHAR_SET.index(ch) if ch in CHAR_SET else len(CHAR_SET)
                                 for ch in word.text][:max_word_len]
            char_indices[-min(len(word.text), max_word_len):] = word_char_indices
            word_char_vectors.append(char_indices)
        return lemmas_vector, grammemes_vector, word_char_vectors

    def __iter__(self):
        """
        Получение очередного батча.
        
        :return: индексы словоформ, грамматические векторы, ответы-индексы.
        """
        for filename in self.filenames:
            yield self.__parse_file(filename)

    def __parse_file(self, filename: str):
        sentences = [[]]
        with tqdm_open(filename, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                word_form = self.__parse_line(line)
                sentences[-1].append(word_form)
                if word_form == SEQ_END_WF:
                    sentences.append([])
                if len(sentences) >= self.batch_size:
                    sentences, next_words = self.__generate_seqs(sentences)
                    yield self.__to_tensor(sentences, next_words)
                    sentences = [[]]

    def __parse_line(self, line: str) -> WordForm:
        line = line.strip()
        if len(line) != 0:
            word, lemma, pos, tags = line.split('\t')[:4]
            word, lemma = word.lower(), lemma.lower() + '_' + pos
            gram_vector_index = self.grammeme_vectorizer.get_index_by_name(pos + "#" + tags)
            return WordForm(lemma, gram_vector_index, word)
        return SEQ_END_WF
