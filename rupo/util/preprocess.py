# -*- coding: utf-8 -*-
# Автор: Гусев Илья
# Описание: Служебные функции и константы.

import re

CYRRILIC_LOWER_VOWELS = "аоэиуыеёюя"
CYRRILIC_LOWER_CONSONANTS = "йцкнгшщзхъфвпрлджчсмтьб"
VOWELS = "aeiouAEIOUаоэиуыеёюяАОЭИУЫЕЁЮЯ"
CLOSED_SYLLABLE_CHARS = "рлймнРЛЙМН"


def text_to_wordlist(sentence, cyrillic=False):
    regexp = "[^а-яА-Яёa-zA-Z]"
    if cyrillic:
        regexp = "[^а-яА-Яё]"
    sentence = re.sub(regexp, " ", sentence)
    result = sentence.lower().split()
    return result


def text_to_sentences(text):
    regexp = "[\.\?!](?=[\s\n]*[A-ZА-Я])|;|:-|:—|:—|: —|: —|: -"
    regexps = ["(?<=[^A-zА-я][A-ZА-Я])\.", 
                "(?<=[^A-zА-я][A-zА-я])\.[ ]?(?=[A-zА-я][^A-zА-я])",
                "\.(?=,)"
                ]
    for reg in regexps:
        text = "$".join(re.split(reg,text))

    result = re.split(regexp, text)
    result = map(lambda x: x.strip().replace("$", "."), result)
    return result


def to_cyrrilic(text):
    return text.replace("x", "х") \
        .replace("a", "а") \
        .replace("y", "у") \
        .replace("o", "о") \
        .replace("c", "с") \
        .replace("ё", "е")


def normilize_line(text):
    regexp = "[^а-яА-Яёa-zA-Z0-9]"
    text = re.sub(regexp, " ", text)
    result = to_cyrrilic("".join(text.lower().split()))
    return result


def count_vowels(string):
    num_vowels = 0
    for char in string:
        if char in VOWELS:
            num_vowels += 1
    return num_vowels


def get_first_vowel_position(string):
    for i in range(len(string)):
        if string[i] in VOWELS:
            return i
    return -1


def etree_to_dict(t):
    return {t.tag: map(etree_to_dict, t.iterchildren()) or t.text}