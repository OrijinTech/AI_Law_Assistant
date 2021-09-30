import sys
from Keyword_dictionary import *
from Error_detector import *


def test_sortfile():
    print("start sorting")
    sort_file_by_pinyin("Dictionaries/Keyword_notebook", "Dictionaries/Sorted_Dictionary")


def test_make_correction(message):
    print("Correct sentence is: ",make_correction(message))

