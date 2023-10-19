import os

from . import six
from .lib_loader import grass2, DocSegCallbackType


class DocSegParser(object):
    def __init__(self, feature_file, dict_file):
        if not os.path.exists(feature_file): raise FileNotFoundError(feature_file)
        if not os.path.exists(dict_file): raise FileNotFoundError(dict_file)

        if isinstance(feature_file, six.text_type):
            feature_file = feature_file.encode("UTF-8")

        if isinstance(dict_file, six.text_type):
            dict_file = dict_file.encode("UTF-8")

        assert isinstance(feature_file, six.binary_type)
        assert isinstance(dict_file, six.binary_type)
        self.parser = grass2.create_docseg_parser(feature_file, dict_file)

    def seg_doc(self, input_str):
        result = []

        def callback(words, length):
            result.extend(words[i].decode("UTF-8") for i in range(length))

        if isinstance(input_str, six.text_type):
            input_str = input_str.encode("UTF-8")

        grass2.parse_string_with_docseg_parser(self.parser, input_str,
                                               DocSegCallbackType(callback))
        return result

    def __del__(self):
        grass2.delete_docseg_parser(self.parser)
