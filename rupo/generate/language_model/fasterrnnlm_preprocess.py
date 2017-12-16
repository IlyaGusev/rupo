from rupo.main.tokenizer import SentenceTokenizer, Tokenizer, Token
import os
from collections import Counter
DATA_DIR = "/media/yallen/My Passport/Datasets/LM/"


def preprocess(input_dir, output_filename, ran, border=60):
    with open(output_filename, 'w', encoding='utf-8') as w:
        for i in ran:
            print(i)
            input_filename = input_dir + str(i) + ".txt"
            if not os.path.exists(input_filename):
                continue
            with open(input_filename, 'r', encoding='utf-8') as r:
                for line in r:
                    sentences = SentenceTokenizer().tokenize(line)
                    for sentence in sentences:
                        tokens = Tokenizer().tokenize(sentence, remove_punct=True, remove_unknown=True, replace_numbers=True)
                        tokens = [token.text.lower() for token in tokens if token.token_type == Token.TokenType.WORD]
                        if len(tokens) > border:
                            continue
                        if len(tokens) > 1:
                            w.write(" ".join(tokens) + "\n")


def shrink_by_vocab(n, input_filename, output_filename):
    counter = Counter()
    with open(input_filename, "r", encoding='utf-8') as r:
        for line in r:
            for word in line.strip().split():
                counter[word] += 1
    most_common = counter.most_common(n)
    words = {word: count for word, count in most_common}
    print(most_common[-1])
    with open(input_filename, "r", encoding='utf-8') as r:
        with open(output_filename, "w", encoding='utf-8') as w:
            for line in r:
                w.write(" ".join([word if word in words else "<UKN>" for word in line.strip().split()]) + "\n")

if __name__ == "__main__":
    preprocess(DATA_DIR, DATA_DIR+ "train-100.txt", list(range(101)))
    shrink_by_vocab(500000, DATA_DIR + "train-100.txt", DATA_DIR + "train-100-ukn.txt")