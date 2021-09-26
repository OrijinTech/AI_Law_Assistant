# 这个文件是用来整理，编辑关键词词典的
import jieba
from pypinyin import *


def dictionary_setup(filename):
    jieba.load_userdict(filename)

def add_to_dictionary(new_word, filename):
    write_to = open(filename, "a", encoding="utf-8")
    jieba.add_word(new_word)
    write_to.write(new_word + "\n")
    write_to.close()

def delete_from_dictionary(word_for_deletion, filename):
    write_to = open(filename, "a", encoding="utf-8")
    jieba.del_word(word_for_deletion)
    write_to.write(word_for_deletion + "\n")
    write_to.close()


#这里需要改进
def sort_pinyin_list(pinyin_list, word_list):
    # Getting the length of list
    # of tuples
    n = len(pinyin_list)
    for i in range(n):
        for j in range(n - i - 1):
            if pinyin_list[j][0] > pinyin_list[j + 1][0]:
                pinyin_list[j], pinyin_list[j + 1] = pinyin_list[j + 1], pinyin_list[j]
                word_list[j], word_list[j + 1] = word_list[j + 1], word_list[j]
    return word_list


def sort_file_by_pinyin(file_to_sort, target_dictionary):
    file_to_work = open(file_to_sort, "r", encoding="utf-8")
    file_word_list = []
    pinyin_list = []
    for line in file_to_work:
        file_word_list.append(line.split(" ", 2))
    print(file_word_list)
    for i in file_word_list:
        pinyin_list.append(lazy_pinyin(i, strict=False))
    print(pinyin_list)
    words_sorted = sort_pinyin_list(pinyin_list, file_word_list)
    print(words_sorted)
    update_dictionary(target_dictionary, words_sorted)



def clear_dictionary(file_to_clear):
    file_to_clear = open(file_to_clear, "w", encoding="utf-8")
    file_to_clear.truncate()


def update_dictionary(sorted_dictionary, list_of_words):
    sorted_to_dictionary = open(sorted_dictionary, "a", encoding="utf-8")
    clear_dictionary(sorted_dictionary)
    for words in list_of_words:
        sorted_to_dictionary.write(words[0] + " " + words[1])






