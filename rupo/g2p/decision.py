# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Получение транскрипции на основе дерева решений.

import os
from sklearn.externals import joblib
from typing import Tuple, List, Set
from sklearn.model_selection import ShuffleSplit, cross_val_score
from rupo.settings import G2P_RU_DICT_PATH, G2P_CLASSIFIER_DIR
from sklearn.tree import DecisionTreeClassifier


class DecisionPhonemePredictor:
    russian_map = {
        "а": ["ɐ", "a", "ɑ", "ə", "æ"],
        "б": ["b", "p"],
        "в": ["f", "v"],
        "г": ["g", "ɡ", "ɣ", "v", "x"],
        "д": ["d", "t"],
        "е": ["ɛ", "e", "ə", "ɪ", "ɨ", "j", "ʝ"],
        "ё": ["ɵ"],
        "ж": ["ʂ", "ɕ", "ʐ", "ʑ"],
        "з": ["ɕ", "s", "z", "ʑ"],
        "и": ["i"],
        "й": ["j"],
        "к": ["k"],
        "л": ["ɫ"],
        "м": ["m", "ɱ"],
        "н": ["n", "ɲ"],
        "о": ["ɐ", "o", "ə"],
        "п": ["p"],
        "р": ["r", "ɾ"],
        "с": ["s", "ɕ", "z"],
        "т": ["t"],
        "у": ["u", "ʉ", "ʊ"],
        "ф": ["f"],
        "х": ["ɣ", "x"],
        "ц": ["ʦ"],
        "ч": ["ʂ", "ɕ", "ʧ", "ʨ", "ɕ"],
        "ш": ["ʂ", "ʧ"],
        "щ": ["ɕ", "ʑ"],
        "ь": ["ʲ"],
        "ы": ["ɨ"],
        "ъ": [],
        "э": ["ɛ", "ɪ"],
        "ю": ["ʉ", "ʊ", "j", "ʝ"],
        "я": ["ə", "æ", "ɪ", "j", "ʝ"]
    }

    russian_alphabet = " абвгдеёжзийклмнопрстуфхцчшщьыъэюя-"
    phonetic_alphabet = " n̪ʃʆäʲ。ˌʰʷːːɐaɑəæbfv̪gɡxtdɛ̝̈ɬŋeɔɘɪjʝɵʂɕʐʑijkјɫlmɱnoprɾszᵻuʉɪ̯ʊɣʦʂʧʨɨɪ̯̯ɲʒûʕχѝíʌɒ‿͡"
    clf_filename = "g2p_clf.pickle"

    def __init__(self) -> None:
        if not os.path.exists(G2P_CLASSIFIER_DIR):
            os.mkdir(G2P_CLASSIFIER_DIR)
        clf_path = os.path.join(G2P_CLASSIFIER_DIR, self.clf_filename)
        if not os.path.isfile(clf_path):
            self.__build_classifier(G2P_RU_DICT_PATH, clf_path)
        self.classifier = joblib.load(clf_path)

    def predict(self, word: str) -> str:
        """
        :param word: слово, для которого надо предугадать транскрипцию.
        :return: транскрипция в МФА.
        """
        samples = DecisionPhonemePredictor.__generate_samples(word)[0]
        answers = self.classifier.predict(samples)
        return "".join([DecisionPhonemePredictor.phonetic_alphabet[i] for i in answers])

    def do_cross_val(self) -> None:
        """
        Кросс-валидация классификатора.
        """
        train_data, train_answers = DecisionPhonemePredictor.__prepare_data(G2P_RU_DICT_PATH)
        cv = ShuffleSplit(2, test_size=0.2, random_state=10)
        cv_scores = cross_val_score(self.classifier, train_data, train_answers, cv=cv, scoring='accuracy')
        print("Cross-validation g2p: " + str(cv_scores.mean()))

    @staticmethod
    def __build_classifier(dict_filename: str, clf_path: str) -> None:
        """
        Постройка классификатора.

        :param dict_filename: путь к файлу словаря.
        :param clf_path: путь, куда сохраняем классификатор.
        """
        train_data, train_answers = DecisionPhonemePredictor.__prepare_data(dict_filename)
        clf = DecisionTreeClassifier()
        clf.fit(train_data, train_answers)
        joblib.dump(clf, clf_path)
        print("Built g2p classifier.")

    @staticmethod
    def __prepare_data(dict_filename: str, context: int=4) -> Tuple[List[List[int]], List[int]]:
        """
        Подготовка данных по словарю.

        :param dict_filename: путь к файлу словаря.
        :return: данные для обучения и ответы к ним.
        """
        clean = []
        with open(dict_filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                graphemes = line.split("\t")[0].strip().lower()
                phonemes = line.split("\t")[1].strip()
                clean.append((graphemes, phonemes))
        train_data = []
        train_answers = []
        for i, (graphemes, phonemes) in enumerate(clean):
            g, p = DecisionPhonemePredictor.__align_phonemes(graphemes, phonemes)
            samples, answers = DecisionPhonemePredictor.__generate_samples(g, p, context)
            train_data += samples
            train_answers += answers
            if i % 10000 == 0:
                print(str(i)+"/"+str(len(clean)))
        return train_data, train_answers

    @staticmethod
    def __generate_samples(graphemes: str, phonemes: str=None, context: int=4) -> Tuple[List[List[int]], List[int]]:
        """
        :param graphemes: слово.
        :param phonemes: транскрипция.
        :return: примеры и ответы для обучения по слову.
        """
        if phonemes is not None:
            assert len(graphemes) == len(phonemes)
        samples = []
        answers = []
        alphabet = "абвгдеёжзийклмнопрстуфхцчшщжъьэюя "
        context = list(range(-context, context+1))
        for i in range(len(graphemes)):
            sample = []
            for c in context:
                if i+c < 0 or i+c >= len(graphemes):
                    for ch in alphabet:
                        sample.append(0)
                else:
                    for ch in alphabet:
                        sample.append(int(graphemes[i+c] == ch))
            samples.append(sample)
            if phonemes is not None:
                answers.append(DecisionPhonemePredictor.phonetic_alphabet.find(phonemes[i]))
        return samples, answers

    @staticmethod
    def __align_phonemes(graphemes: str, phonemes: str) -> Tuple[str, str]:
        """
        Выравнивание графем и фонем.

        :param graphemes: графемы.
        :param phonemes: фонемы.
        :return: выровненная пара.
        """
        diff = len(graphemes) - len(phonemes)
        phonemes_variants = DecisionPhonemePredictor.__alignment_variants(phonemes, diff, set()) \
            if diff > 0 else [phonemes]
        graphemes_variants = DecisionPhonemePredictor.__alignment_variants(graphemes, abs(diff), set()) \
            if diff < 0 else [graphemes]
        scores = {}
        for g in graphemes_variants:
            for p in phonemes_variants:
                assert len(g) == len(p)
                scores[(g, p)] = DecisionPhonemePredictor.__score_alignment(g, p)
        return max(scores, key=scores.get)

    @staticmethod
    def __alignment_variants(symbols: str, space_count: int, spaces: Set[str]) -> Set[str]:
        """
        Получение вариантов выравнивания.

        :param symbols: буквы.
        :param space_count: количество пробелов, которые осталось расставить
        :param spaces: позиции пробелов.
        :return: варианты выравнивания.
        """
        if space_count == 0:
            answer = ""
            next_symbol = 0
            for i in range(len(symbols) + len(spaces)):
                if i in spaces:
                    answer += " "
                else:
                    answer += symbols[next_symbol]
                    next_symbol += 1
            return {answer}
        variants = set()
        for j in range(len(symbols) + space_count):
            if j not in spaces:
                variants |= DecisionPhonemePredictor.__alignment_variants(symbols, space_count - 1, spaces | {j})
        return variants

    @staticmethod
    def __score_alignment(graphemes: str, phonemes: str) -> int:
        """
        Оценка выравнивания.

        :param graphemes: графемы.
        :param phonemes: фонемы.
        :return: оценка.
        """
        score = 0
        for i in range(len(graphemes)):
            grapheme = graphemes[i]
            phoneme = phonemes[i]
            if phoneme == " ":
                if i-1 >= 0 and phonemes[i-1] in DecisionPhonemePredictor.russian_map[grapheme]:
                    score += 0.5
            elif grapheme == " ":
                if i+1 < len(graphemes) and graphemes[i+1] != " " and \
                                phoneme in DecisionPhonemePredictor.russian_map[graphemes[i+1]]:
                    score += 0.5
            elif phoneme in DecisionPhonemePredictor.russian_map[grapheme]:
                score += 1
        return score

