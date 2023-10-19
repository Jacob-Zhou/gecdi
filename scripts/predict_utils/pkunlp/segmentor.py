import os

from . import six
from .lib_loader import grass, UTF8


class Segmentor(object):
    def __init__(self, feature_file, dict_file):
        if not os.path.exists(feature_file): raise FileNotFoundError(feature_file)
        if not os.path.exists(dict_file): raise FileNotFoundError(dict_file)

        if isinstance(feature_file, six.text_type):
            feature_file = feature_file.encode("UTF-8")

        if isinstance(dict_file, six.text_type):
            dict_file = dict_file.encode("UTF-8")
        self.segmentor = grass.create_segmentor_ctx(feature_file, dict_file)

    def seg_string(self, input_str, encoding="UTF-8"):
        if isinstance(input_str, six.text_type):
            input_str = input_str.encode(encoding)
        return [i.decode(encoding)
                for i in grass.seg_string_with_ctx(
                self.segmentor, input_str, UTF8).split(b"\x09")]

    def __del__(self):
        if hasattr(self, "segmentor"):
            grass.delete_segmentor_ctx(self.segmentor)
