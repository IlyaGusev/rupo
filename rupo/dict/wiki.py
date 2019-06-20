import re

from rupo.settings import RU_GRAPHEME_SET

from russ.syllables import VOWELS


class WikiDict:
    @staticmethod
    def convert_to_g2p_only(source_file, destination_file):
        with open(source_file, 'r', encoding='utf-8') as r:
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
    def first_clean_up(filename):
        words = []
        phonetic_words = []
        with open(filename, "r") as f:
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
                        sum([1 for ch in phonetic_word if ch in VOWELS]):
                    continue
                words.append(word)
                phonetic_words.append(phonetic_word)
        print(len(words))
        with open(filename, "w") as f:
            for i, word in enumerate(words):
                f.write(word + "\t" + phonetic_words[i] + "\n")