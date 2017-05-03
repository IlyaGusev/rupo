# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Конвертер CMU словаря.

from rupo.settings import CMU_DICT, EN_PHONEME_STRESS_PATH, EN_G2P_DICT_PATH


class CMUDict:
    aprabet2ipa = {
        "AO": "ɔ",
        "AA": "ɑ",
        "IY": "i",
        "UW": "u",
        "EH": "ɛ",
        "IH": "ɪ",
        "UH": "ʊ",
        "AH": "ʌ",
        "AX": "ə",
        "AE": "æ",
        "EY": "eɪ",
        "AY": "aɪ",
        "OW": "oʊ",
        "AW": "aʊ",
        "OY": "ɔɪ",
        "ER": "ɝ",
        "AXR": "ɚ",
        "P": "p",
        "B": "b",
        "T": "t",
        "D": "d",
        "K": "k",
        "G": "ɡ",
        "CH": "ʦ",
        "JH": "ʤ",
        "F": "f",
        "V": "v",
        "TH": "θ",
        "DH": "ð",
        "S": "s",
        "Z": "z",
        "SH": "ʃ",
        "ZH": "ʒ",
        "HH": "h",
        "M": "m",
        "EM": "m",
        "N": "n",
        "EN": "n",
        "NG": "ŋ",
        "ENG": "ŋ",
        "L": "ɫ",
        "EL": "ɫ",
        "R": "r",
        "DX": "ɾ",
        "NX": "ɾ",
        "Y": "j",
        "W": "w",
        "Q": "ʔ"
    }

    diphtongs = ["EY", "AY", "OW", "AW", "OY"]

    @staticmethod
    def convert_to_g2p_only(destination_file):
        clean = []
        with open(CMU_DICT, 'r', encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            for line in lines:
                g = line.split("  ")[0].lower()
                if not ("a" <= g[0] <= "z"):
                    continue
                if "(" in g:
                    continue
                p = line.split("  ")[1].strip()
                phonemes = p.split(" ")
                for i, phoneme in enumerate(phonemes):
                    if not ("A" <= phoneme[-1] <= "Z"):
                        phonemes[i] = phoneme[:-1]
                p = "".join([CMUDict.aprabet2ipa[phoneme] for phoneme in phonemes])
                clean.append((g, p))
        with open(destination_file, 'w', encoding="utf-8") as w:
            for g, p in clean:
                w.write(g+"\t"+p+"\n")

    @staticmethod
    def convert_to_phoneme_stress(destination_file):
        clean = []
        with open(CMU_DICT, 'r', encoding="utf-8", errors="ignore") as f:
            for line in f:
                g = line.split("  ")[0].lower()
                if not ("a" <= g[0] <= "z"):
                    continue
                p = line.split("  ")[1].strip()
                if "(1)" in g:
                    g = g.replace("(1)", "")
                if "(2)" in g:
                    g = g.replace("(2)", "")
                if "(" in g:
                    continue

                phonemes = p.split(" ")
                primary = []
                secondary = []
                diphtongs_count = 0
                for i, phoneme in enumerate(phonemes):
                    if not ("A" <= phoneme[-1] <= "Z"):
                        if int(phoneme[-1]) == 1:
                            primary.append(str(i+diphtongs_count))
                        if int(phoneme[-1]) == 2:
                            secondary.append(str(i+diphtongs_count))
                        phonemes[i] = phoneme[:-1]
                        if phonemes[i] in CMUDict.diphtongs:
                            diphtongs_count += 1
                p = "".join([CMUDict.aprabet2ipa[phoneme] for phoneme in phonemes])
                clean.append((p, primary, secondary))
        with open(destination_file, 'w', encoding="utf-8") as w:
            for p, f, s in clean:
                w.write(p + "\t" + ",".join(f) + "\t" + ",".join(s) + "\n")
# if __name__ == '__main__':
#     CMUDict.convert_to_phoneme_stress(EN_PHONEME_STRESS_PATH)
#     CMUDict.convert_to_g2p_only(EN_G2P_DICT_PATH)