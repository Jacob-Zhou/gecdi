from . import six
from .lib_loader import grass, UTF8


class SemanticParser(object):
    def __init__(self, semantic_feature_file, tree_feature_file):
        self.parser = grass.create_semantic_parser_ctx(semantic_feature_file, tree_feature_file)

    def parse(self, sentence, encoding="UTF-8"):
        if isinstance(sentence, six.text_type):
            sentence = sentence.encode(encoding)
        elif isinstance(sentence, list):
            sentence = " ".join(six.text_type(i) for i in sentence).encode(encoding)
        return grass.semantic_parse_string_with_ctx(self.parser, sentence, UTF8).decode(encoding)