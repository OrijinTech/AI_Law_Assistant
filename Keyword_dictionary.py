# 这个文件是用来整理，编辑关键词词典的
import jieba
from pypinyin import *
from support_fnc import *


def dictionary_setup(filename):
    '''
    初始化结巴分词的内置词典工具
    :param filename: 词典文件的文件名或路径。词典格式：txt。注：此词典文件为处理前的词典原文件。
    :type filename: file
    :return: 无
    :rtype: 无
    '''
    jieba.load_userdict(filename)

def add_to_dictionary(new_word, filename):
    '''
    结巴分词内置function，用来添加新词汇至词典。
    :param new_word: 要添加的词汇
    :type new_word: str
    :param filename: 添加目的地文件
    :type filename: file
    :return:无
    :rtype:无
    '''
    write_to = open(filename, "a", encoding="utf-8")
    jieba.add_word(new_word)
    write_to.write(new_word + "\n")
    write_to.close()

def delete_from_dictionary(word_for_deletion, filename):
    '''
    从词典中删除特定词语。
    :param word_for_deletion: 删除的词语
    :type word_for_deletion: str
    :param filename:词汇所在文件
    :type filename:file
    :return:无
    :rtype:无
    '''
    write_to = open(filename, "a", encoding="utf-8")
    jieba.del_word(word_for_deletion)
    write_to.write(word_for_deletion + "\n")
    write_to.close()


#这里需要改进
def sort_pinyin_list(pinyin_list, word_list):
    '''
    分词Function，把一个list里的词语进行排列。
    :param pinyin_list: 词语拼音的list（拼音）
    :type pinyin_list:list
    :param word_list: 词语的list（汉字）
    :type word_list: list
    :return:汉字的list
    :rtype:list
    '''
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
    '''
    分词Function，把一个词典文件里的所有词汇进行排列，并添加到一个新的词典文件里。排列顺序为拼音的A-Z字母顺序。
    :param file_to_sort: 排序的文件
    :type file_to_sort: file
    :param target_dictionary: 排序过后的文件
    :type target_dictionary: file
    :return:无
    :rtype:无
    '''
    file_to_work = open(file_to_sort, "r", encoding="utf-8")
    file_word_list = []
    pinyin_list = []
    for line in file_to_work:
        file_word_list.append(line.split(" ", 2))
    #print(file_word_list)
    for i in file_word_list:
        pinyin_list.append(lazy_pinyin(i, strict=False))
    #(pinyin_list)
    words_sorted = sort_pinyin_list(pinyin_list, file_word_list)
    #print(words_sorted)
    update_dictionary(target_dictionary, words_sorted)


def clear_dictionary(file_to_clear):
    '''
    清空一个词典内的所有词语。
    :param file_to_clear:要清空的文件
    :type file_to_clear: file
    :return:无
    :rtype:无
    '''
    file_to_clear = open(file_to_clear, "w", encoding="utf-8")
    file_to_clear.truncate()


def update_dictionary(sorted_dictionary, list_of_words):
    '''
    把词汇写入新的词典。
    :param sorted_dictionary:词典文件
    :type sorted_dictionary: file
    :param list_of_words: 输入到新词典的词语list
    :type list_of_words: list
    :return:无
    :rtype:无
    '''
    sorted_to_dictionary = open(sorted_dictionary, "a", encoding="utf-8")
    clear_dictionary(sorted_dictionary)
    for words in list_of_words:
        if cont_num(words):
            sorted_to_dictionary.write(words[0] + " " + words[1])
        else:
            sorted_to_dictionary.write(words[0] + " ")


def search_keyword(dictionary, keyword):
    '''
    在词典里搜索对应的词语 https://stackoverflow.com/questions/3449384/fastest-text-search-method-in-a-large-text-file
    :param dictionary: 目标搜索词典
    :type dictionary: file
    :param keyword: 需要搜索的关键词
    :type keyword: str
    :return: str or none for not found
    :rtype: str
    '''
    file_to_work = open(dictionary, "r", encoding="utf-8")
    is_found = ""
    print("正在寻找词语:", keyword)
    for each_word in file_to_work:
        if len(each_word) == 0:
            break
        if keyword in each_word:
            is_found = each_word
            break
    return is_found


def in_dictionary(dictionary, keyword):
    '''
    词典里是否存在该词语
    :param dictionary: 目标搜索词典
    :type dictionary: file
    :param keyword: 需要搜索的关键词
    :type keyword: str
    :return: 该词典（有/没有）词语
    :rtype: bool
    '''
    cmpword = search_keyword(dictionary, keyword)
    if keyword in cmpword:
        return True
    else:
        print("未在对应词典内找到您所需要的关键词")
        return False


