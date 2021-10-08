# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
from Word_splitter import *
from Keyword_dictionary import *
from Error_detector import *
from test_file import *
import filecmp
from os.path import exists

user_message = ""


def get_user_message():
    user_problem = input("你遇到了什么法律问题?\n请输入：")
    return user_problem


# Set up everything here
def machine_set_up():
    setup_jieba()
    # if two dictionaries are different
    if exists("Dictionaries/Keyword_notebook") and exists("Dictionaries/Sorted_Dictionary"):
        if not filecmp.cmp("Dictionaries/Keyword_notebook", "Dictionaries/Sorted_Dictionary"):
            sort_file_by_pinyin("Dictionaries/Keyword_notebook", "Dictionaries/Sorted_Dictionary")
        dictionary_setup("Dictionaries/Sorted_Dictionary")
    else:
        print("未检测到关键词词典，请把txt格式的词典放入文件夹内。")


def main(start_var):
    machine_set_up()
    test_search_keyword()
    test_in_dictionary()
    # message_text = "猫猫太可爱怎么办，可以偷吗？会不会要赔猫，怎么赔？" #get_user_message()
    # corrected_text = make_correction(message_text)
    # split_list = fen_ci(corrected_text, True)
    # print(split_list)
    # add_to_dictionary(split_list[0], "Dictionaries/Keyword_notebook")
    return 0


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])
