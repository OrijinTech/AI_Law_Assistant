import sys
from Keyword_dictionary import *


def start(start_var):
    print("start sorting")
    sort_file_by_pinyin("Dictionaries/Keyword_notebook", "Dictionaries/Sorted_Dictionary")

start(sys.argv[0])


