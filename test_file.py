import sys
from Keyword_dictionary import *
from Error_detector import *


def test_sortfile():
    print("start sorting")
    sort_file_by_pinyin("Dictionaries/Keyword_notebook", "Dictionaries/Sorted_Dictionary")


def test_make_correction(message):
    print("Correct sentence is: ", make_correction(message))


def test_search_keyword():
    print("已找到词语:", search_keyword("Dictionaries/Test_dictionary", "抢劫"))


def test_in_dictionary():
    print("该词语在词典中：", in_dictionary("Dictionaries/Test_dictionary","抢劫"))