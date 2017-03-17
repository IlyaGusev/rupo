from pkg_resources import resource_filename
foo_config = resource_filename(__name__, 'foo.conf')

DICT_TXT_PATH = resource_filename(__name__, "data/dict/accents_dict.txt")
DICT_TRIE_PATH = resource_filename(__name__, "data/dict/accents_dict.trie")
CLASSIFIER_PATH = resource_filename(__name__, "data/classifier/")