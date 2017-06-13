class Phonemes:
    VOWELS = ["i", "y", "ɪ", "ʏ", "ɨ", "ʉ", "ʊ", "ɯ", "u", "e", "ø", "ɘ", "ɵ", "ɤ", "o",
              "ə", "ɛ", "œ", "ɜ", "ɞ", "ʌ", "ɔ", "æ", "ɐ", "a", "ɶ", "ä", "ɑ", "ɒ", "ɝ",
              "ɚ"]
    APPROXIMANT_CONSONANTS = ["ʋ", "ɹ", "ɻ", "j", "ɰ", "ʍ", "w", "ɥ", "ʕ", "ʁ",
                              "l", "ɭ", "ʟ", "ʎ", "ɫ"]
    NASAL_CONSONANTS = ["m", "ɱ", "n", "ɳ", "ɲ", "ŋ", "ɴ", ]
    FLAP_OR_TAP_CONSONANTS = ["ⱱ", "ɾ", "ɽ", "ɺ"]
    IMPLOSIVE_CONSONANTS = ["ɓ", "ɗ", "ʄ", "ɠ", "ʛ"]
    CLICK_CONSONANTS = ["ʘ", "ǀ", "ǃ", "ǂ", "ǁ"]
    AFFRICATE_CONSONANTS = ["ʦ", "ʣ", "ʧ", "ʤ", "ʨ", "ʥ"]
    STOP_CONSONANTS = ["p", "b", "t", "d", "ʈ", "ɖ", "c", "ɟ", "k", "ɡ", "q",
                       "ɢ", "ʡ", "ʔ"]
    FRICATIVE_CONSONANTS = ["s", "z", "ʃ", "ʒ", "ʂ", "ʐ", "ɕ", "ʑ", "ɸ", "β",
                            "f", "v", "θ", "ð", "ç", "ʝ", "x", "ɣ", "χ", "ħ",
                            "h", "ɦ", "ɬ", "ɮ", "ʩ"]
    TRILL_CONSONANTS = ["ʙ", "r", "ʀ", "ʜ", "ʢ"]
    DIACRITICS = ["ʲ", "̯"]
    SUPRASEGMENTALS = ["ː"]
    LIGATURES = {
        "t͡s": "ʦ",
        "d͡z": "ʣ",
        "t͡ʃ": "ʧ",
        "d͡ʒ": "ʤ",
        "t͡ɕ": "ʨ",
        "d͡ʑ": "ʥ"
    }

    @staticmethod
    def get_all():
        l = [" "] + Phonemes.VOWELS + Phonemes.NASAL_CONSONANTS + Phonemes.STOP_CONSONANTS + \
            Phonemes.FRICATIVE_CONSONANTS + Phonemes.AFFRICATE_CONSONANTS + Phonemes.APPROXIMANT_CONSONANTS + \
            Phonemes.FLAP_OR_TAP_CONSONANTS + Phonemes.TRILL_CONSONANTS + Phonemes.CLICK_CONSONANTS + \
            Phonemes.IMPLOSIVE_CONSONANTS + Phonemes.DIACRITICS + Phonemes.SUPRASEGMENTALS
        return l

    @staticmethod
    def clean(phonemes: str) -> str:
        for letters, ch in Phonemes.LIGATURES.items():
            phonemes = phonemes.replace(letters, ch)
        clean = ""
        alphabet = Phonemes.get_all()
        for i, ch in enumerate(phonemes):
            if ch == "ː":
                if phonemes[i-1] in Phonemes.VOWELS:
                    clean += phonemes[i-1]
                continue
            if ch in alphabet + ["'", "ˌ"]:
                clean += ch
        j_positions = [i for i in range(len(clean)) if clean.startswith("ɪ̯", i)]
        offset = 0
        for pos in j_positions:
            if not (pos-1 >= 0 and clean[pos-1] not in Phonemes.VOWELS
                    and pos+2 < len(clean) and clean[pos+2] not in Phonemes.VOWELS):
                clean = clean[:pos-offset] + "j" + clean[pos-offset+2:]
                offset += 1
        return clean

    @staticmethod
    def get_sonority(phonemes: str):
        inverse_sonority_levels = [(Phonemes.VOWELS, 0), (Phonemes.APPROXIMANT_CONSONANTS, 1),
                                   (Phonemes.NASAL_CONSONANTS, 2), (Phonemes.FLAP_OR_TAP_CONSONANTS, 2),
                                   (Phonemes.IMPLOSIVE_CONSONANTS, 2), (Phonemes.STOP_CONSONANTS, 3),
                                   (Phonemes.CLICK_CONSONANTS, 3), (Phonemes.AFFRICATE_CONSONANTS, 4),
                                   (Phonemes.FRICATIVE_CONSONANTS, 5), (Phonemes.TRILL_CONSONANTS, 2)]
        sonority = []
        for ch in phonemes:
            for chars, level in inverse_sonority_levels:
                if ch in chars:
                    sonority.append(level)
            if ch in Phonemes.DIACRITICS or ch in Phonemes.SUPRASEGMENTALS:
                sonority.append(-1)
        return sonority

    @staticmethod
    def get_syllables(phonemes: str):
        sonority = Phonemes.get_sonority(phonemes)
        vowels_positions = [i for i, level in enumerate(sonority) if level == 0]
        intervals = [(vowels_positions[i]+1, vowels_positions[i+1]) for i in range(len(vowels_positions)-1)]
        borders = []
        for begin, end in intervals:
            sonority_interval = sonority[begin: end]
            border = begin
            if len(sonority_interval) != 0:
                border += len(sonority_interval) - sonority_interval[-1::-1].index(max(sonority_interval)) - 1
            borders.append(border)
        if len(borders) == 0:
            borders = [0]
        if borders[0] != 0:
            borders = [0] + borders
        if borders[-1] != len(phonemes):
            borders += [len(phonemes)]
        return [phonemes[borders[i]:borders[i+1]] for i in range(len(borders)-1)]