from pkg_resources import resource_filename
foo_config = resource_filename(__name__, 'foo.conf')

DICT_TXT_PATH = resource_filename(__name__, "data/dict/accents_dict.txt")
DICT_TRIE_PATH = resource_filename(__name__, "data/dict/accents_dict.trie")
CLASSIFIER_PATH = resource_filename(__name__, "data/classifier/")
MARKUP_XML_EXAMPLE = resource_filename(__name__, "data/examples/markup.xml")
MARKUP_JSON_EXAMPLE = resource_filename(__name__, "data/examples/markup.json")
TEXT_XML_EXAMPLE = resource_filename(__name__, "data/examples/text.xml")
TEXT_TXT_EXAMPLE = resource_filename(__name__, "data/examples/text.txt")
EXAMPLES_DIR = resource_filename(__name__, "data/examples/")