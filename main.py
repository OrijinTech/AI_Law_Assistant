# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
from Word_splitter import *
from Keyword_dictionary import *

user_message = ""


def get_user_message():
    user_problem = input("你遇到了什么法律问题?\n请输入：")
    return user_problem


#Set up everything here
def machine_set_up():
    setup_jieba()
    dictionary_setup("Keyword_notebook")


def main(start_var):
    machine_set_up()
    message_text = get_user_message()
    add_to_dictionary("我")
    split_list = fen_ci(message_text, True)
    print(split_list)
    add_to_dictionary(split_list[0])


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])
