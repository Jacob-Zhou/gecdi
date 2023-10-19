import os
import sys
import ctypes
import platform


def detect_lib_name(project_name):
    """
    Select corresponding libgrass
    :return: library file name
    """
    if sys.platform in ['win32', 'cygwin']:
        system = "win"
        ext = 'dll'
    elif sys.platform.startswith("linux"):
        system = "linux"
        ext = 'so'
    else:
        raise SystemError("Your system ({}) is not supported.".format(sys.platform))
    bit = platform.architecture()[0][:2]  # 32 or 64
    return "lib{}-{}{}.{}".format(project_name, system, bit, ext)


lib_file = os.path.join(os.path.dirname(__file__), "lib", detect_lib_name("grass"))
lib2_file = os.path.join(os.path.dirname(__file__), "lib", detect_lib_name("grass2"))


class CWordWithTag(ctypes.Structure):
    _fields_ = [("word", ctypes.c_char_p), ("tag", ctypes.c_char_p)]


class TaggingResult(ctypes.Structure):
    _fields_ = [("words", ctypes.POINTER(CWordWithTag)),
                ("length", ctypes.c_int)]


grass = ctypes.cdll.LoadLibrary(lib_file)
grass2 = ctypes.cdll.LoadLibrary(lib2_file)

DocSegCallbackType = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char_p),
                                      ctypes.c_size_t)

grass2.create_docseg_parser.restype = ctypes.c_longlong
grass2.parse_string_with_docseg_parser.argtypes = (ctypes.c_longlong, ctypes.c_char_p,
                                                   DocSegCallbackType)

NERCallbackType = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)
grass2.create_ner_parser.restype = ctypes.c_longlong
grass2.parse_string_with_ner_parser.argtypes = (ctypes.c_longlong, ctypes.c_char_p,
                                                NERCallbackType)

POSTaggerCallbackType = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)
grass2.libgrass_create_postagger.restype = ctypes.c_longlong
grass2.libgrass_run_tagger.argtypes = (ctypes.c_longlong, ctypes.c_char_p,
                                                POSTaggerCallbackType)

grass.seg_string_with_ctx.restype = ctypes.c_char_p
grass.tag_sentence_with_ctx.restype = TaggingResult
grass.syntax_parse_string_with_ctx.restype = ctypes.c_char_p
grass.semantic_parse_string_with_ctx.restype = ctypes.c_char_p
UTF8 = 65001
