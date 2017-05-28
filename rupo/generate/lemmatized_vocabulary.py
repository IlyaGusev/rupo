# -*- coding: utf-8 -*-
import numpy as np


class LemmatizedWord(object):
    def __init__(self, lemma, gr_tag, word_form):
        self.lemma = lemma
        self.gr_tag = gr_tag
        self.word_form = word_form

    def __repr__(self):
        return "<Lemma = {}; GrTag = {}; WordForm = {}>".format(self.lemma, self.gr_tag, self.word_form)

    def __eq__(self, other):
        return (self.lemma, self.gr_tag, self.word_form) == (other.lemma, other.gr_tag, other.word_form)

    def __hash__(self):
        return hash((self.lemma, self.gr_tag, self.word_form))


class LemmatizedVocabulary(object):
    def __init__(self, voc_path, lemmas_counter=None, dictionary=None):
        self.word_form2_lemmatization = {}
        self.lemmatization2word_form = {}
        self.lemmatizedWords = []
        self.lemma2lemmatizedWord = {}
        self.lemmatizedWordIndicies = {}
        if voc_path:
            with open(voc_path, encoding='utf8') as f:
                for line in f:
                    lemma, gr_tag, word_form = line.strip().split(' ')
                    gr_tag = int(gr_tag)
                    self.lemmatizedWords.append(LemmatizedWord(lemma, gr_tag, word_form))
                    if word_form not in self.word_form2_lemmatization:
                        self.word_form2_lemmatization[word_form] = []
                    self.word_form2_lemmatization[word_form].append((lemma, gr_tag))
                    if (lemma, gr_tag) not in self.lemmatization2word_form:
                        self.lemmatization2word_form[(lemma, gr_tag)] = []
                    self.lemmatization2word_form[(lemma, gr_tag)].append(word_form)
                    if lemma not in self.lemma2lemmatizedWord:
                        self.lemma2lemmatizedWord[lemma] = []
                    self.lemma2lemmatizedWord[lemma].append(LemmatizedWord(lemma, gr_tag, word_form))
        else:
            for pair in dictionary:
                if pair[0] not in self.lemma2lemmatizedWord:
                    self.lemma2lemmatizedWord[pair[0]] = []
                self.lemma2lemmatizedWord[pair[0]].append(LemmatizedWord(pair[0], pair[1], dictionary[pair]))
            self.word_form2_lemmatization = {dictionary[pair] : LemmatizedWord(pair[0], pair[1], dictionary[pair]) 
                                             for pair in dictionary}
            self.lemmatization2word_form = {self.word_form2_lemmatization[x] : x for x in self.word_form2_lemmatization}
            for lemma, _ in lemmas_counter.most_common():
                self.lemmatizedWords.extend(self.lemma2lemmatizedWord[lemma])
        self.lemmatizedWordIndicies = {x : i for i, x in enumerate(self.lemmatizedWords)}
        self.index2lemmatizedWord = {i : x for i, x in enumerate(self.lemmatizedWords)}
                
    def get_word_form(self, lemma, gr_tag):
        return self.lemmatization2word_form[(lemma, gr_tag)] \
                if (lemma, gr_tag) in self.lemmatization2word_form[(lemma, gr_tag)] \
                else None
    
    def get_lemmatization(self, word_form):
        return self.word_form2_lemmatization[word_form] \
                if word_form in self.word_form2_lemmatization[word_form] \
                else None
    
    def choice_word(self):
        return self.lemmatizedWords[np.random.randint(0, len(self.lemmatizedWords))]
    
    def get_paradigm(self, lemma):
        return self.lemma2lemmatizedWord[lemma] if lemma in self.lemma2lemmatizedWord else None
    
    def get_word_form_index(self, lemmatizedWord):
        return self.lemmatizedWordIndicies[lemmatizedWord] \
            if lemmatizedWord in self.lemmatizedWordIndicies \
            else len(self.lemmatizedWordIndicies)
            
    def get_word(self, index):
        return self.index2lemmatizedWord[index]