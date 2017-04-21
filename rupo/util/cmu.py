# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Конвертер CMU словаря.

from rupo.settings import CMU_DICT, EN_PHONEME_ACCENT_PATH, EN_G2P_DICT_PATH


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
        "CH": "tʃ",
        "JH": "dʒ",
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
        "EM": "m̩",
        "N": "n",
        "EN": "n̩",
        "NG": "ŋ",
        "ENG": "ŋ̍",
        "L": "ɫ",
        "EL": "ɫ̩",
        "R": "r",
        "DX": "ɾ",
        "NX": "ɾ̃",
        "Y": "j",
        "W": "w",
        "Q": "ʔ"
    }

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
    def convert_to_phoneme_accent(destination_file):
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
                primary = -1
                secondary = -1
                for i, phoneme in enumerate(phonemes):
                    if not ("A" <= phoneme[-1] <= "Z"):
                        if int(phoneme[-1]) == 1:
                            primary = i
                        if int(phoneme[-1]) == 2:
                            secondary = i
                        phonemes[i] = phoneme[:-1]
                p = "".join([CMUDict.aprabet2ipa[phoneme] for phoneme in phonemes])
                clean.append((p, primary, secondary))
        with open(destination_file, 'w', encoding="utf-8") as w:
            for p, f, s in clean:
                w.write(p + "\t" + str(f) + "\t" + str(s) + "\n")
