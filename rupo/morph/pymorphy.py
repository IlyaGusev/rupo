import pymorphy2
from russian_tagsets import converters
from rupo.main.tokenizer import Tokenizer, Token


class RussianMorphology:
    @staticmethod
    def do_markup(input_filename, output_filename):
        morph = pymorphy2.MorphAnalyzer()
        to_ud = converters.converter('opencorpora-int', 'ud14')
        with open(input_filename, "r", encoding="utf-8") as inp:
            with open(output_filename, "w", encoding="utf-8") as out:
                for line in inp:
                    tokens = Tokenizer.tokenize(line)
                    accepted_types = [Token.TokenType.WORD]
                    tokens = [token for token in tokens if token.token_type in accepted_types]
                    for token in tokens:
                        text = token.text.lower()
                        parse = morph.parse(text)[0]
                        lemma = parse.normal_form
                        ud_tag = to_ud(str(parse.tag), text)
                        pos = ud_tag.split()[0]
                        gram = ud_tag.split()[1]
                        out.write("%s\t%s\t%s\t%s\n" % (text, lemma, pos, gram))
                    out.write("\n")





