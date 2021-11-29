import sys
from Tools.Keyword_dictionary import *
from Tools.Error_detector import *
from Tools.data_modification import *


def test_sortfile():
    print("start sorting")
    sort_file_by_pinyin("../Dictionaries/Keyword_notebook", "Dictionaries/Sorted_Dictionary")


def test_make_correction(message):
    print("Correct sentence is: ", make_correction(message))


def test_search_keyword():
    print("已找到词语:", search_keyword("Dictionaries/Test_dictionary", "抢劫"))


def test_in_dictionary():
    print("该词语在词典中：", in_dictionary("Dictionaries/Test_dictionary","抢劫"))


def test_data_modification(sentence, law_type):
    add_to_file(sentence, law_type)
    print("Now look at Law_Data.json file to see the update!")

