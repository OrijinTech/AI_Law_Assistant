import json
from difflib import SequenceMatcher
import jieba
import nltk
import numpy as np
import pycorrector
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


def open_file(file_name, report):
    if report == "Y":
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
def bag_of_words_ch(s, words):
    # # print("Inside bag_of_words_chinese function.")
    # stop_words = set(stopwords(["zh"]))
    # bag_chinese = [0 for _ in range(len(words))]
    # s_words = jieba.cut(s, cut_all=False)
    # # stopwords filtration
    # s_words_filtered = []
    # for w in s_words:
    #     if w not in stop_words:
    #         s_words_filtered.append(w)
    # # unicode_s_words_filtered = [i.decode('utf-8') for i in s_words_filtered]
    # for se in s_words_filtered:
    #     for i, w in enumerate(words):
    #         if w == se:
    #             bag_chinese[i] = 1
    # return np.array(bag_chinese)
    # print("Inside bag_of_words_chinese function.")

    bag_chinese = [0 for _ in range(len(words))]
    s_words = jieba.cut(s, cut_all=False)
    s_words = list(s_words)
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag_chinese[i] = 1
    return np.array(bag_chinese)


def bag_of_words_en(s, words):
    stemmer = LancasterStemmer()
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


# return a sentence with the specific word filtered out
def word_filter_rem(sentence, word):
    word_list = jieba.cut(sentence)
    updated_sent = ""
    for w in word_list:
        if w == word:
            word_list.remove(w)
    for st in word_list:
        updated_sent += st
    return updated_sent


def report_intent(results, results_index, intent_type):
    result_percentage = round(results[results_index] * 100, 2)
    if results[results_index] > 0.7:
        message = "AIYU有 {} % 的确定性,这是 {}.".format(result_percentage, intent_type)
    else:
        message = "很抱歉, 我只有 {}% 的确定性.".format(result_percentage)
    return message


def split_sentence(sentence, language):
    tokenized_list_of_words = []
    if language == "ch":
        # if len(sentence) > 5:
        #     sentence = word_filter_rem(sentence, "您好")
        sentence_cut = jieba.cut(sentence, cut_all=False)
        tokenized_list_of_words = list(sentence_cut)
    elif language == "en":
        tokenized_list_of_words = nltk.word_tokenize(sentence)
    else:
        if language != "en" or language != "ch":
            print("Cannot identify the correct language to split the sentence.")
        elif sentence.empty():
            print("Cannot identify the sentence.")
        else:
            print("Unknown error during split_sentence.")
    return tokenized_list_of_words

def add_pattern(sentence, law_type):
    with open("D:\AI_Law_Assistant\Language_Data\Law_Data.json", "r", encoding="utf8") as file:
        # returning the Json object
        data = json.load(file)
        for intent in data["law_database"]:
            # For each pattern inside each intent, we want to tokenize the words that are in sentences.
            if intent["law_type"] == law_type:
                intent["law_keywords"].append(sentence)
    with open("D:\AI_Law_Assistant\Language_Data\Law_Data.json", "w", encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

def make_correction(message):
    '''
    寻找并纠正用户信息内的错误。
    :param message: 用户输入信息
    :type message: str（中文）
    :return: 纠正过后的用户信息
    :rtype: str（中文）
    '''
    correct_message, detail = pycorrector.correct(message)
    # print(correct_message)
    return correct_message