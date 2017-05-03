from pkg_resources import resource_filename
foo_config = resource_filename(__name__, 'foo.conf')

CLASSIFIER_DIR = resource_filename(__name__, "data/classifier/")

CMU_DICT = resource_filename(__name__, "data/dict/cmu.txt")
ZALIZNYAK_DICT = resource_filename(__name__, "data/dict/zaliznyak.txt")
RU_WIKI_DICT = resource_filename(__name__, "data/dict/wiki_ru.txt")

RU_GRAPHEME_STRESS_PATH = resource_filename(__name__, "data/dict/ru_grapheme_stress.txt")
RU_GRAPHEME_STRESS_TRIE_PATH = resource_filename(__name__, "data/dict/ru_grapheme_stress.trie")
RU_G2P_DICT_PATH = resource_filename(__name__, "data/dict/ru_g2p.txt")
RU_PHONEME_STRESS_PATH = resource_filename(__name__, "data/dict/ru_phoneme_stress.txt")
RU_PHONEME_STRESS_BIG_PATH = resource_filename(__name__, "data/dict/ru_phoneme_stress_big.txt")
RU_PHONEME_STRESS_TRIE_PATH = resource_filename(__name__, "data/dict/ru_phoneme_stress.trie")
RU_G2P_DEFAULT_MODEL = resource_filename(__name__, "data/g2p_models/g2p_ru_maxlen35_LSTM256-BLSTM256_LSTM128_dropout0.2_acc99_wer9.h5")
RU_STRESS_DEFAULT_MODEL = resource_filename(__name__, "data/stress_models/stress_ru_LSTM128_dropout0.2_acc99_wer0.h5")

EN_G2P_DICT_PATH = resource_filename(__name__, "data/dict/en_g2p.txt")
EN_PHONEME_STRESS_PATH = resource_filename(__name__, "data/dict/en_phoneme_stress.txt")
EN_PHONEME_STRESS_TRIE_PATH = resource_filename(__name__, "data/dict/en_phoneme_stress.trie")
EN_G2P_DEFAULT_MODEL = resource_filename(__name__, "data/g2p_models/g2p_en_maxlen30_LSTM256_LSTM128_dropout0.2_acc96_wer26.h5")
EN_STRESS_DEFAULT_MODEL = resource_filename(__name__, "data/stress_models/stress_en_LSTM128_dropout0.2_acc99_wer10.h5")

EXAMPLES_DIR = resource_filename(__name__, "data/examples/")
MARKUP_XML_EXAMPLE = resource_filename(__name__, "data/examples/markup.xml")
MARKUP_JSON_EXAMPLE = resource_filename(__name__, "data/examples/markup.json")
TEXT_XML_EXAMPLE = resource_filename(__name__, "data/examples/text.xml")
TEXT_TXT_EXAMPLE = resource_filename(__name__, "data/examples/text.txt")
HYPHEN_TOKENS = resource_filename(__name__, "data/hyphen-tokens.txt")

G2P_CURRENT_MODEL_DIR = resource_filename(__name__, "data/g2p_models/")
ACCENT_CURRENT_MODEL_DIR = resource_filename(__name__, "data/stress_models/")

TEMP_PATH = resource_filename(__name__, "data/temp.txt")

RU_GRAPHEME_SET = " абвгдеёжзийклмнопрстуфхцчшщьыъэюя-"
EN_GRAPHEME_SET = " abcdefghijklmnopqrstuvwxyz.'-"