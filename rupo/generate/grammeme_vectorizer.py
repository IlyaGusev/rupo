import pickle
from collections import defaultdict
from rupo.settings import GENERATOR_GRAM_VECTORS


def get_empty_category():
    return {GrammemeVectorizer.UNKNOWN_VALUE}


class GrammemeVectorizer:
    UNKNOWN_VALUE = "Unknown"

    def __init__(self, dump_filename=GENERATOR_GRAM_VECTORS):
        self.all_grammemes = defaultdict(get_empty_category)
        self.vectors = []
        self.name_to_index = {}
        self.dump_filename = dump_filename

    def save(self) -> None:
        with open(self.dump_filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.dump_filename, "rb") as f:
            vocab = pickle.load(f)
            self.__dict__.update(vocab.__dict__)

    def collect_grammemes(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                pos_tag, grammemes = line.split("\t")[2:4]
                self.all_grammemes["POS"].add(pos_tag)
                grammemes = grammemes.split("|") if grammemes != "_" else []
                for grammeme in grammemes:
                    category = grammeme.split("=")[0]
                    value = grammeme.split("=")[1]
                    self.all_grammemes[category].add(value)

    def collect_possible_vectors(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if line == "\n":
                    continue
                pos_tag, grammemes = line.split("\t")[2:4]
                vector_name = pos_tag + "#" + grammemes
                if vector_name not in self.name_to_index:
                    grammemes = grammemes.split("|") if grammemes != "_" else []
                    vector = self.__build_vector(pos_tag, grammemes)
                    self.vectors.append(vector)
                    self.name_to_index[vector_name] = len(self.vectors) - 1

    def get_vector(self, vector_name: str):
        if vector_name not in self.name_to_index:
            raise RuntimeError("Unknown POS tag and grammemes combination")
        return self.vectors[self.name_to_index[vector_name]]

    def get_ordered_grammemes(self):
        flat = []
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            for value in sorted(list(values)):
                flat.append(category+"="+value)
        return flat

    def grammemes_count(self):
        return len(self.get_ordered_grammemes())

    def __build_vector(self, pos_tag, grammemes):
        vector = []
        gram_tags = {pair.split("=")[0]: pair.split("=")[1] for pair in grammemes}
        gram_tags["POS"] = pos_tag
        sorted_grammemes = sorted(self.all_grammemes.items(), key=lambda x: x[0])
        for category, values in sorted_grammemes:
            if category not in gram_tags:
                vector += [1 if value == GrammemeVectorizer.UNKNOWN_VALUE else 0 for value in sorted(list(values))]
            else:
                vector += [1 if value == gram_tags[category] else 0 for value in sorted(list(values))]
        return vector
