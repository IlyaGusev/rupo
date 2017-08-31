class UDConverter:
    @staticmethod
    def convert_from_conllu(input_filename, output_filename, with_forth_column=False):
        with open(input_filename, "r") as r:
            with open(output_filename, "w") as w:
                for line in r:
                    if line[0] == "#" or line[0] == "=":
                        continue
                    if line != "\n":
                        records = line.split("\t")
                        if with_forth_column:
                            grammems = records[5].strip().split("|")
                        else:
                            grammems = records[4].strip().split("|")
                        dropped = ["Animacy", "Aspect", "NumType"]
                        grammems = [grammem for grammem in grammems if sum([drop in grammem for drop in dropped ]) == 0]
                        grammems = "|".join(grammems)
                        pos = records[3]
                        if pos != "PUNCT":
                            w.write("\t".join([records[1], records[2].lower(), pos, grammems]) + "\n")
                    else:
                        w.write("\n")


# import os
# dir_name = "/media/data/Datasets/Morpho"
# UDConverter.convert_from_conllu(os.path.join(dir_name, "gikrya_fixed.txt"),
#                                 os.path.join(dir_name, "clean", "gikrya.txt"), with_forth_column=False)
# UDConverter.convert_from_conllu(os.path.join(dir_name, "RNCgoldInUD_Morpho.conll"),
#                                 os.path.join(dir_name, "clean", "rnc.txt"), with_forth_column=False)
# UDConverter.convert_from_conllu(os.path.join(dir_name, "ru-ud-train.conllu"),
#                                 os.path.join(dir_name, "clean", "ru-ud-train.txt"), with_forth_column=True)
# UDConverter.convert_from_conllu(os.path.join(dir_name, "syntagrus_full.ud"),
#                                 os.path.join(dir_name, "clean", "syntagrus.txt"), with_forth_column=False)
# UDConverter.convert_from_conllu(os.path.join(dir_name, "unamb_sent_14_6.conllu"),
#                                 os.path.join(dir_name, "clean", "opencorpora.txt"), with_forth_column=True)
