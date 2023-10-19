from __future__ import unicode_literals
import ctypes
from collections import namedtuple

import os

from . import six
from . import six_plus
from .lib_loader import grass2, POSTaggerCallbackType


@six_plus.python_2_unicode_compatible
class WordWithTag(namedtuple("WordWithTag", ["word", "tag"])):
    def __str__(self):
        return "{}/{}".format(self.word, self.tag)

    def __repr__(self):
        return str(self)


class POSTaggerV2(object):
    def __init__(self, feature_file):
        if isinstance(feature_file, six.text_type):
            feature_file = feature_file.encode("UTF-8")
        if not os.path.exists(feature_file): raise FileNotFoundError(feature_file)

        self.tagger = grass2.libgrass_create_postagger(feature_file)

    def tag_sentence(self, words):
        input_string = b"\x03".join([i.encode("UTF-8") for i in words])
        result = []

        def callback(word, tag):
            result.append(WordWithTag(word.decode("UTF-8"), tag.decode("UTF-8")))

        grass2.libgrass_run_tagger(self.tagger, input_string,
                                   POSTaggerCallbackType(callback))
        return result

    def __del__(self):
        if hasattr(self, "tagger"):
            grass2.libgrass_delete_postagger(self.tagger)
