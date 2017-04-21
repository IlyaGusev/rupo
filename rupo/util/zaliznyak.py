import os

from rupo.g2p.rnn_g2p import RNNPhonemePredictor
from rupo.g2p.aligner import Aligner
from rupo.settings import RU_G2P_DEFAULT_MODEL, ZALIZNYAK_DICT, TEMP_PATH, RU_GRAPHEME_ACCENT_PATH, RU_G2P_DICT_PATH


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
        g2p_predictor.build()
        g2p_predictor.load(RU_G2P_DEFAULT_MODEL)
        with open(ZALIZNYAK_DICT, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(destination_file, 'w', encoding='utf-8') as w:
            for line in lines:
                for word in line.split("#")[1].split(","):
                    word = word.strip()
                    clean_word = ""
                    for i, ch in enumerate(word):
                        if ch == "'" or ch == "`":
                            continue
                        clean_word += ch
                    w.write(clean_word + "\t" + g2p_predictor.predict(word) + "\n")

    @staticmethod
    def convert_to_phoneme_accent(destination_file):
        g2p_predictor = RNNPhonemePredictor()
        g2p_predictor.build()
        g2p_predictor.load(RU_G2P_DEFAULT_MODEL)
        ZalyzniakDict.convert_to_accent_only(TEMP_PATH)
        with open(TEMP_PATH, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(destination_file, 'w', encoding='utf-8') as w:
            for line in lines:
                word, primary, secondary = line.split("\t")
                primary = int(primary)
                secondary = int(secondary)
                if primary == -1:
                    continue
                phonemes = g2p_predictor.predict(word)
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
                            assert phonemes[new_primary] != " "
                        if old_pos == secondary:
                            new_secondary = old_pos + spaces_count
                            assert phonemes[new_secondary] != " "
                        old_pos += 1
                assert new_primary != -1
                print(g, p, new_primary, new_secondary)
                w.write(phonemes + "\t" + str(new_primary) + "\t" + str(new_secondary) + "\n")
        os.remove(TEMP_PATH)