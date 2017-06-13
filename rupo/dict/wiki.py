import re

from rupo.settings import RU_G2P_DICT_PATH, RU_PHONEME_STRESS_PATH, RU_WIKI_DICT, RU_GRAPHEME_SET
from rupo.g2p.phonemes import Phonemes


class WikiDict:
    @staticmethod
    def convert_to_g2p_only(destination_file):
        with open(RU_WIKI_DICT, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(destination_file, 'w', encoding='utf-8') as w:
            words = []
            phonetic_words = []
            for line in lines:
                words.append(line.split("\t")[0].strip())
                phonetic_words.append(line.split("\t")[1].replace("'", "").replace("ˌ", "").strip())
            for i, word in enumerate(words):
                w.write(word + "\t" + phonetic_words[i] + "\n")

    @staticmethod
    def convert_to_phoneme_stress(destination_file):
        with open(RU_WIKI_DICT, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(destination_file, 'w', encoding='utf-8') as w:
            for line in lines:
                word = line.split("\t")[1].strip()
                clean_word = word.replace("'", "").replace("ˌ", "")
                all_stresses = [i for i, ch in enumerate(word) if ch == "'" or ch == "ˌ"]
                primary = [i for i, ch in enumerate(word) if ch == "'"]
                secondary = [i for i, ch in enumerate(word) if ch == "ˌ"]
                primary_stresses = []
                secondary_stresses = []
                for pos in primary:
                    stress = -1
                    for i, ch in enumerate(word[pos:]):
                        if ch in Phonemes.VOWELS:
                            stress = i + pos - all_stresses.index(pos) - 1
                            break
                    if stress != -1:
                        primary_stresses.append(str(stress))
                for pos in secondary:
                    stress = -1
                    for i, ch in enumerate(word[pos:]):
                        if ch in Phonemes.VOWELS:
                            stress = i + pos - all_stresses.index(pos) - 1
                            break
                    if stress != -1:
                        secondary_stresses.append(str(stress))
                w.write(clean_word + "\t" + ",".join(primary_stresses) + "\t" + ",".join(secondary_stresses) + "\n")

    @staticmethod
    def first_clean_up():
        words = []
        phonetic_words = []
        with open(RU_WIKI_DICT, "r") as f:
            lines = f.readlines()
            print(len(lines))
            for line in lines:
                word = line.split("#")[0]
                word = word.lower()
                phonetic_word = line.split("#")[1]
                if "'" not in phonetic_word and "ˈ" not in phonetic_word:
                    continue
                phonetic_word = phonetic_word.split("/")[0].strip()
                phonetic_word = phonetic_word.split("~")[0].strip()
                phonetic_word = phonetic_word.split(";")[0].strip()
                phonetic_word = phonetic_word.split(",")[0].strip()
                phonetic_word = phonetic_word.replace("ˈ", "'")
                phonetic_word = phonetic_word.replace(":", "ː")
                phonetic_word = re.sub(r"[\s̟̥̻.̞]", "", phonetic_word)
                phonetic_word = re.sub(r"[(⁽][^)⁾]*[)⁾]", "", phonetic_word)
                phonetic_word = Phonemes.clean(phonetic_word)
                wrong_chars = [ch for ch in word if ch not in RU_GRAPHEME_SET]
                if len(wrong_chars) != 0:
                    continue
                if len(word) == 0 or len(phonetic_word) == 0:
                    continue
                if sum([1 for ch in word if ch in "еуаоэяиюёы"]) != \
                        sum([1 for ch in phonetic_word if ch in Phonemes.VOWELS]):
                    continue
                words.append(word)
                phonetic_words.append(phonetic_word)
        print(len(words))
        with open(RU_WIKI_DICT, "w") as f:
            for i, word in enumerate(words):
                f.write(word + "\t" + phonetic_words[i] + "\n")


# if __name__ == '__main__':
    # WikiDict.first_clean_up()
    # WikiDict.convert_to_g2p_only(RU_G2P_DICT_PATH)
    # WikiDict.convert_to_phoneme_stress(RU_PHONEME_STRESS_PATH)