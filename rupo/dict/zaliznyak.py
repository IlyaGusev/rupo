import os

from rupo.g2p.rnn_g2p import RNNPhonemePredictor
from rupo.g2p.aligner import Aligner
from rupo.settings import RU_G2P_DEFAULT_MODEL, ZALIZNYAK_DICT, TEMP_PATH, RU_G2P_DICT_PATH, RU_PHONEME_STRESS_PATH


class ZalyzniakDict:
    @staticmethod
    def convert_to_accent_only(destination_file):
        with open(ZALIZNYAK_DICT, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(destination_file, 'w', encoding='utf-8') as w:
            for line in lines:
                for word in line.split("#")[1].split(","):
                    word = word.strip()
                    pos = -1
                    clean_word = ""
                    primary = []
                    secondary = []
                    for i, ch in enumerate(word):
                        if ch == "'" or ch == "`":
                            if ch == "`":
                                secondary.append(pos)
                            else:
                                primary.append(pos)
                            continue
                        clean_word += ch
                        pos += 1
                        if ch == "ё":
                            primary.append(pos)
                    if len(primary) != 0:
                        w.write(clean_word + "\t" + ",".join([str(a) for a in primary]) + "\t" +
                                ",".join([str(a) for a in secondary]) + "\n")

    @staticmethod
    def convert_to_g2p_only(destination_file):
        g2p_predictor = RNNPhonemePredictor()
        g2p_predictor.load(RU_G2P_DEFAULT_MODEL)
        with open(ZALIZNYAK_DICT, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(destination_file, 'w', encoding='utf-8') as w:
            words = []
            for line in lines:
                for word in line.split("#")[1].split(","):
                    word = word.strip()
                    clean_word = ""
                    for i, ch in enumerate(word):
                        if ch == "'" or ch == "`":
                            continue
                        clean_word += ch
                    words.append(clean_word)
            phonetic_words = g2p_predictor.predict(words)
            for i, word in enumerate(words):
                w.write(word + "\t" + phonetic_words[i] + "\n")

    @staticmethod
    def convert_to_phoneme_stress(destination_file):
        g2p_predictor = RNNPhonemePredictor()
        g2p_predictor.load(RU_G2P_DEFAULT_MODEL)
        ZalyzniakDict.convert_to_accent_only(TEMP_PATH)
        vowels = ["ɐ", "a", "ɑ", "ə", "æ", "ɛ", "e", "ɪ", "ɨ", "j", "ʝ", "ɵ", "i", "o", "u", "ʉ", "ʊ", "ɨ",
                  "ɛ̝̈", "ä", "ɔ", "ᵻ", "ɪ", "ɪ̯̯", "û", "ɒ", "ː"]
        with open(TEMP_PATH, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(destination_file, 'w', encoding='utf-8') as w:
            for line in lines:
                word, p, s = line.split("\t")
                primary = int(p.strip().split(",")[0])
                if s.strip() != "":
                    secondary = int(s.strip().split(",")[0])
                else:
                    secondary = -1
                if primary == -1:
                    continue
                phonemes = g2p_predictor.predict([word])[0]
                if abs(len(phonemes)-len(word)) > 4:
                    continue
                g, p = Aligner.align_phonemes(word, phonemes)
                new_primary = -1
                new_secondary = -1
                old_pos = 0
                spaces_count = 0
                for i in range(len(g)):
                    if p[i] == " ":
                        spaces_count -= 1
                    if g[i] == " ":
                        spaces_count += 1
                    else:
                        if old_pos == primary:
                            new_primary = old_pos + spaces_count
                        if old_pos == secondary and secondary != -1:
                            new_secondary = old_pos + spaces_count
                        old_pos += 1
                assert new_primary != -1
                if phonemes[new_primary] == "ʲ":
                    new_primary += 1
                if new_primary >= len(phonemes):
                    continue
                if phonemes[new_primary] in vowels:
                    print(g, p, new_primary, new_secondary)
                    w.write(phonemes + "\t" + str(new_primary) + "\t" + str(new_secondary) + "\n")
                else:
                    print("Failed on: " + phonemes[new_primary] + " " + p + " " + g)
        os.remove(TEMP_PATH)

if __name__ == '__main__':
    ZalyzniakDict.convert_to_phoneme_stress(RU_PHONEME_STRESS_PATH)