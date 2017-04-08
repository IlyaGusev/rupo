from typing import Tuple, List


class MLPhonemeClassifier:
    russian_map = {
        "а": ["ɐ", "a", "ɑ", "ə", "æ"],
        "б": ["b", "bʲ", "p", "pʲ"],
        "в": ["f", "fʲ", "v", "vʲ"],
        "г": ["g", "ɡʲ", "ɣ", "v", "x", "xʲ"],
        "д": ["d", "dʲ", "t", "tʲ"],
        "е": ["ɛ", "e", "ə", "ɪ", "ɨ", "j", "ʝ"],
        "ё": ["ɵ"],
        "ж": ["ʂ", "ɕː", "ʐ", "ʑː"],
        "з": ["ɕː", "s", "sʲ", "z", "zʲ", "ʑː"],
        "и": ["i"],
        "й": ["j"],
        "к": ["k", "kʲ"],
        "л": ["ɫ", "lʲ"],
        "м": ["m", "mʲ", "ɱ"],
        "н": ["n", "nʲ"],
        "о": ["ɐ", "o", "ə"],
        "п": ["p", "pʲ"],
        "р": ["r", "ɾ", "rʲ", "ɾʲ", "r。", "rʲ。"],
        "с": ["s", "sʲ", "ɕː", "zʲ"],
        "т": ["t", "tʲ"],
        "у": ["u", "ʉ", "ʊ"],
        "ф": ["f", "fʲ"],
        "х": ["ɣ", "x", "xʲ"],
        "ц": ["ʦ"],
        "ч": ["ʂ", "ɕː", "ʧ", "ʨ"],
        "ш": ["ʂ", "ʧ"],
        "щ": ["ɕː", "ʑː"],
        "ь": ["bʲ", "ɡʲ", "dʲ", "kʲ", "lʲ", "mʲ", "nʲ", "pʲ", "sʲ", "fʲ", "tʲ", "zʲ", "vʲ", "xʲ"],
        "ы": ["ɨ"],
        "ъ": [],
        "э": ["ɛ", "ɪ"],
        "ю": ["ʉ", "ʊ", "j", "ʝ"],
        "я": ["ə", "æ", "ɪ", "j", "ʝ"]
    }

    @staticmethod
    def generate_g2p_samples(graphemes, phonemes) -> Tuple[List[List[int]], List[int]]:
        samples = []
        answers = []
        alphabet = "абвгдеёжзийклмнопрстуфхцчшщжъьэюя "
        context = list(range(-4, 5))
        for i in range(len(graphemes)):
            sample = []
            for c in context:
                if i+c < 0 or i+c >= len(graphemes):
                    for ch in alphabet:
                        sample.append(False)
                else:
                    for ch in alphabet:
                        sample.append(graphemes[i+c] == ch)
            samples.append(sample)
            answers.append(phonemes[i])
        return samples, answers

    @staticmethod
    def align_phonemes(graphemes, phonemes):
        diff = len(graphemes) - len(phonemes)
        phonemes_variants = MLPhonemeClassifier.alignment_variants(phonemes, diff, set()) if diff > 0 else [phonemes]
        graphemes_variants = MLPhonemeClassifier.alignment_variants(graphemes, abs(diff), set()) if diff < 0 else [graphemes]
        scores = {}
        for g in graphemes_variants:
            for p in phonemes_variants:
                assert len(g) == len(p)
                scores[(g, p)] = MLPhonemeClassifier.score_alignment(g, p)
        return max(scores, key=scores.get)

    @staticmethod
    def alignment_variants(symbols, space_count, spaces):
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
                variants |= MLPhonemeClassifier.alignment_variants(symbols, space_count-1, spaces | {j})
        return variants

    @staticmethod
    def score_alignment(graphemes, phonemes):
        score = 0
        for i in range(len(graphemes)):
            grapheme = graphemes[i]
            phoneme = phonemes[i]
            if phoneme == " ":
                if i-1 >= 0 and phonemes[i-1] in MLPhonemeClassifier.russian_map[grapheme]:
                    score += 0.5
            elif grapheme == " ":
                if i+1 < len(graphemes) and phonemes[i] in MLPhonemeClassifier.russian_map[graphemes[i+1]]:
                    score += 0.5
            elif phoneme in MLPhonemeClassifier.russian_map[grapheme]:
                score += 1
        return score