from rupo.main.markup import Markup, Line, Word, Syllable

MARKUP_EXAMPLE = Markup("Соломка изжила себя.\n Пора виться майкой в.", [
            Line(0, 21, "Соломка изжила себя.", [
                Word(0, 7, "Соломка",
                     [Syllable(0, 2, 0, "Со"),
                      Syllable(2, 5, 1, "лом", 3),
                      Syllable(5, 7, 2, "ка")]),
                Word(8, 14, "изжила",
                     [Syllable(0, 1, 0, "и"),
                      Syllable(1, 4, 1, "зжи"),
                      Syllable(4, 6, 2, "ла", 5)]),
                Word(15, 19, "себя",
                     [Syllable(0, 2, 0, "се"),
                      Syllable(2, 4, 1, "бя", 3)])]),
            Line(21, 43, " Пора виться майкой в.",[
                Word(22, 26, "Пора",
                     [Syllable(0, 2, 0, "По", 1),
                      Syllable(2, 4, 1, "ра", 3)]),
                Word(27, 33, "виться",
                     [Syllable(0, 2, 0, "ви", 1),
                      Syllable(2, 6, 1, "ться")]),
                Word(34, 40, "майкой",
                     [Syllable(0, 3, 0, "май", 1),
                      Syllable(3, 6, 1, "кой")]),
                Word(41, 42, "в", [])
                ])])