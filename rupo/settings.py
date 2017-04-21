from pkg_resources import resource_filename
foo_config = resource_filename(__name__, 'foo.conf')

CLASSIFIER_DIR = resource_filename(__name__, "data/classifier/")

CMU_DICT = resource_filename(__name__, "data/dict/cmu.txt")
ZALIZNYAK_DICT = resource_filename(__name__, "data/dict/zaliznyak.txt")
RU_WIKI_DICT = resource_filename(__name__, "data/dict/ru_wiki.txt")

RU_GRAPHEME_ACCENT_PATH = resource_filename(__name__, "data/dict/ru_grapheme_accent_dict.txt")
RU_GRAPHEME_ACCENT_TRIE_PATH = resource_filename(__name__, "data/dict/ru_grapheme_accent_dict.trie")
RU_G2P_DICT_PATH = RU_WIKI_DICT
RU_PHONEME_ACCENT_PATH = resource_filename(__name__, "data/dict/ru_phoneme_accent_dict.txt")
RU_PHONEME_ACCENT_TRIE_PATH = resource_filename(__name__, "data/dict/ru_phoneme_accent_dict.trie")
RU_G2P_DEFAULT_MODEL = resource_filename(__name__, "data/g2p_models/RU_BLSTM256-DROP20-WER28-EPOCH10.hdf5")

EN_G2P_DICT_PATH = resource_filename(__name__, "data/dict/en_g2p_dict.txt")
EN_PHONEME_ACCENT_PATH = resource_filename(__name__, "data/dict/en_phoneme_accent_dict.txt")
EN_PHONEME_ACCENT_TRIE_PATH = resource_filename(__name__, "data/dict/en_phoneme_accent_dict.trie")
EN_G2P_DEFAULT_MODEL = resource_filename(__name__, "data/g2p_models/EN_BLSTM128-LSTM128-DROP20-WER43-EPOCH14.hdf5")

EXAMPLES_DIR = resource_filename(__name__, "data/examples/")
MARKUP_XML_EXAMPLE = resource_filename(__name__, "data/examples/markup.xml")
MARKUP_JSON_EXAMPLE = resource_filename(__name__, "data/examples/markup.json")
TEXT_XML_EXAMPLE = resource_filename(__name__, "data/examples/text.xml")
TEXT_TXT_EXAMPLE = resource_filename(__name__, "data/examples/text.txt")
HYPHEN_TOKENS = resource_filename(__name__, "data/hyphen-tokens.txt")

G2P_CURRENT_MODEL_DIR = resource_filename(__name__, "data/g2p_models/")
ACCENT_CURRENT_MODEL_DIR = resource_filename(__name__, "data/accent_models/")

TEMP_PATH = resource_filename(__name__, "data/temp.txt")

PHONEME_SET = " n̪ʃʆäʲ。ˌʰʷːːɐaɑəæbfv̪gɡxtdɛ̝̈ɬŋeɔɘɪjʝɵʂɕʐʑijkјɫlmɱnoprɾszᵻuʉɪ̯ʊɣʦʂʧʨɨɪ̯̯ɲʒûʕχѝíʌɒ‿͡ðwhɝθ"
RU_GRAPHEME_SET = " абвгдеёжзийклмнопрстуфхцчшщьыъэюя-"
EN_GRAPHEME_SET = " abcdefghijklmnopqrstuvwxyz.'-"