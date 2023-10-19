import sys
from .six import PY2


def get_default_encoding():
    return sys.stdout.encoding or "utf8"


default_encoding = get_default_encoding()


def python_2_unicode_compatible(klass):
    """
    A decorator that defines __unicode__ and __str__ methods under Python 2.
    Under Python 3 it does nothing.

    To support Python 2 and 3 with a single code base, define a __str__ method
    returning text and apply this decorator to the class.
    """
    if PY2:
        if '__str__' not in klass.__dict__:
            raise ValueError("@python_2_unicode_compatible cannot be applied "
                             "to %s because it doesn't define __str__()." %
                             klass.__name__)
        klass.__unicode__ = klass.__str__
        klass.__str__ = lambda self: self.__unicode__().encode(default_encoding)
    return klass


def fix_encoding():
    if PY2 and sys.stdout.encoding is None:
        sys.stdout.encoding = "utf-8"
