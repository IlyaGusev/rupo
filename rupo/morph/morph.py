import pymorphy2
from russian_tagsets import converters
from rupo.main.tokenizer import Tokenizer, Token


class RussianMorphology:
    @staticmethod
    def pymorphy_process(input_filename, output_filename):
        """
        Сделать морфоразметку на вход генератору с помощью pymorphy2 и russian-tagsets.
        
        :param input_filename: входной файл - raw текст.
        :param output_filename: выходной файл - разметка.
        """
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

    @staticmethod
    def vocab_process(input_filename, output_filename, word_form_vocabulary, grammeme_vectorizer):
        """
        Сделать морфоразметку по словарю словоформ.
        
        :param input_filename: входной файл - raw текст.
        :param output_filename: выходной файл - разметка.
        :param word_form_vocabulary: слоаврь словоформ.
        :param grammeme_vectorizer: векторы граммем.
        :return: 
        """
        with open(input_filename, "r", encoding="utf-8") as r:
            with open(output_filename, "w", encoding="utf-8") as w:
                for line in r:
                    tokens = Tokenizer.tokenize(line)
                    accepted_types = [Token.TokenType.WORD]
                    tokens = [token for token in tokens if token.token_type in accepted_types]
                    for token in tokens:
                        text = token.text.lower()
                        word_forms = word_form_vocabulary.get_word_forms_by_text(text)
                        if len(word_forms) == 0:
                            print(text)
                        else:
                            indices = [word_form_vocabulary.get_word_form_index(form) for form in word_forms]
                            index = min(indices)
                            word_form = word_form_vocabulary.get_word_form_by_index(index)
                            name = grammeme_vectorizer.get_name_by_index(word_form.gram_vector_index)
                            pos = name.split("#")[0]
                            gram = name.split("#")[1]
                            w.write("%s\t%s\t%s\t%s\n" % (text, word_form.lemma, pos, gram))
                    w.write("\n")
