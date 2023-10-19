# encoding: utf-8

from pkunlp import Segmentor
import unittest

class TestSegmentor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSegmentor, self).__init__(*args, **kwargs)
        self.segmentor = Segmentor("feature/segment.feat", "feature/segment.dic")

    def test_punc(self):
        segments = self.segmentor.seg_string("100%(百分百)表示一个数是另一个数的百分之几的数,叫做百分数。")
        self.assertIn("100%", segments)
        self.assertIn("(", segments)
        self.assertIn(")", segments)


if __name__ == '__main__':
    unittest.main()

