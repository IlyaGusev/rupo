import os


class ZalyzniakDict:
    @staticmethod
    def convert_to_accent_only(dict_file, accent_file):
        with open(dict_file, 'r', encoding='utf-8') as r:
            lines = r.readlines()
        with open(accent_file, 'w', encoding='utf-8') as w:
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