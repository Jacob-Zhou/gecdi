import os

from . import six
from .lib_loader import grass2, NERCallbackType


class NERTagger(object):
    def __init__(self, feature_file):
        if not os.path.exists(feature_file): raise FileNotFoundError(feature_file)

        if isinstance(feature_file, six.text_type):
            feature_file = feature_file.encode("UTF-8")

        assert isinstance(feature_file, six.binary_type)
        self.parser = grass2.create_ner_parser(feature_file)

    def tag(self, input_str):
        result = []

        def callback(word, tag):
            result.append((word.decode("UTF-8"), tag.decode("UTF-8")))

        if isinstance(input_str, six.text_type):
            input_str = input_str.encode("UTF-8")

        grass2.parse_string_with_ner_parser(self.parser, input_str,
                                            NERCallbackType(callback))
        return result

    def __del__(self):
        if hasattr(self, "parser"):
            grass2.delete_ner_parser(self.parser)
