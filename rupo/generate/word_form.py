class WordForm(object):
    def __init__(self, lemma, gram_vector_index, text):
        self.lemma = lemma  # type: str
        self.gram_vector_index = gram_vector_index  # type: str
        self.text = text  # type: str

    def __repr__(self):
        return "<Lemma = {}; GrTag = {}; WordForm = {}>".format(self.lemma, self.gram_vector_index, self.text)

    def __eq__(self, other):
        return (self.lemma, self.gram_vector_index, self.text) == (other.lemma, other.gram_vector_index, other.text)

    def __hash__(self):
        return hash((self.lemma, self.gram_vector_index, self.text))