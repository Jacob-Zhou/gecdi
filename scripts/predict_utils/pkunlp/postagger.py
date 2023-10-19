from __future__ import unicode_literals
import ctypes
from collections import namedtuple

import os

from . import six
from . import six_plus
from .lib_loader import grass, UTF8


@six_plus.python_2_unicode_compatible
class WordWithTag(namedtuple("WordWithTag", ["word", "tag"])):
    def __str__(self):
        return "{}/{}".format(self.word, self.tag)

    def __repr__(self):
        return str(self)


class POSTagger(object):
    def __init__(self, feature_file):
        if isinstance(feature_file, six.text_type):
            feature_file = feature_file.encode("UTF-8")
        if not os.path.exists(feature_file): raise FileNotFoundError(feature_file)

        self.tagger = grass.create_postagger_ctx(feature_file)

    def tag_sentence(self, sentences, encoding="UTF-8"):
        if isinstance(sentences, (six.string_types, six.binary_type)):
            sentences = [sentences]

        length = len(sentences)
        array = (ctypes.c_char_p * length)()
        for i in range(length):
            array[i] = sentences[i].encode(encoding)

        result = grass.tag_sentence_with_ctx(self.tagger, array, length, UTF8)
        return [WordWithTag(word=result.words[i].word.decode(encoding), tag=result.words[i].tag.decode(encoding))
                for i in range(result.length)]

    def __del__(self):
        if hasattr(self, "tagger"):
            grass.delete_postagger_ctx(self.tagger)
