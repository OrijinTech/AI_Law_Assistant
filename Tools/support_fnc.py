import json
from difflib import SequenceMatcher
import jieba
import numpy as np
import tflearn
from nltk import LancasterStemmer
from stopwordsiso import stopwords



def cont_num(keyword) -> bool:
    '''
    检测str内是否有integer
    :param keyword: 输入词语
    :type keyword: str
    :return: 有数字或没数字
    :rtype: bool
    '''
    return any(char.isdigit() for char in keyword)


def open_file(file_name):
    print("Opening file: ", file_name)
    # Open the json file with the train data
    with open(file_name, encoding="utf8") as file:
        # returning the Json object
        data = json.load(file)
    return data


def clear_chat():
    for i in range(50):
        print("\n")


def check_similarity(a, b):
    return SequenceMatcher(None, a, b).quick_ratio()


def get_max_similarity_percentage(sentence, list_of_resp):
    responses_prob = []
    for sentences_in_list in list_of_resp:
        similarity_percentage = check_similarity(word_filter_rem(sentence, "您好"), sentences_in_list)
        responses_prob.append(similarity_percentage)
    max_prob = max(responses_prob)
    return responses_prob.index(max_prob)

# SETUP for Chinese sentence processing
def bag_of_words_chinese(s, words):
    # print("Inside bag_of_words_chinese function.")
    stop_words = set(stopwords(["zh"]))
    bag_chinese = [0 for _ in range(len(words))]
    s_words = jieba.cut(s, cut_all=False)
    # stopwords filtration
    s_words_filtered = []
    for w in s_words:
        if w not in stop_words:
            s_words_filtered.append(w)
    # unicode_s_words_filtered = [i.decode('utf-8') for i in s_words_filtered]
    for se in s_words_filtered:
        for i, w in enumerate(words):
            if w == se:
                bag_chinese[i] = 1
        return np.array(bag_chinese)

#return a sentence with the specific word filtered out
def word_filter_rem(sentence, word):
    word_list = jieba.cut(sentence)
    updated_sent = ""
    for w in word_list:
        if w == word:
            word_list.remove(w)
    for st in word_list:
        updated_sent += st
    return updated_sent

