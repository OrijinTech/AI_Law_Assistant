import json
from difflib import SequenceMatcher
import jieba
import nltk
import numpy as np
import pycorrector
from nltk import LancasterStemmer
from datetime import datetime
from stopwordsiso import stopwords  # Might use later


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


def bag_of_words(s, words, language):
    bag_chinese = [0 for _ in range(len(words))]
    s_words = split_sentence(s, language)
    words_list = list(s_words)
    for se in words_list:
        for i, w in enumerate(words):
            if w == se:
                bag_chinese[i] = 1
    return np.array(bag_chinese)


def word_filter_rem(sentence, word):
    word_list = jieba.cut(sentence)
    updated_sent = ""
    for w in word_list:
        if w == word:
            word_list.remove(w)
    for st in word_list:
        updated_sent += st
    return updated_sent


def report_train_results(results, results_index, intent_type, labels):
    result_percentage = round(results[results_index] * 100, 2)
    print("---------REPORTING RESULTS---------"
          "\nProbabilities of Types:", "\n", results,
          "\nCategories:", "\n", labels,
          "\nIndex of the Type with Largest Probability:", results_index,
          "\nIndex of the Resulted Type:", intent_type,
          "\nProbability of the Type:", result_percentage, "%")


def split_sentence(sentence, language):
    tokenized_list_of_words = []
    if language == "ch":
        sentence_cut = jieba.cut(sentence, cut_all=False)
        tokenized_list_of_words = list(sentence_cut)
    elif language == "en":
        stemmer = LancasterStemmer()
        tokenized_list_of_words = nltk.word_tokenize(sentence)
        tokenized_list_of_words = [stemmer.stem(word.lower()) for word in tokenized_list_of_words]
    else:
        if language != "en" or language != "ch":
            print("Cannot identify the correct language to split the sentence.")
        elif sentence.empty():
            print("Cannot identify the sentence.")
        else:
            print("Unknown error during split_sentence.")
    return tokenized_list_of_words


def add_pattern(intent_file, intents, tags, pattern, sentence, learn_type):
    with open(intent_file, "r", encoding="utf8") as file:
        # returning the Json object
        data = json.load(file)
        for intent in data[intents]:
            # For each pattern inside each intent, we want to tokenize the words that are in sentences.
            if intent[tags] == learn_type:
                intent[pattern].append(sentence)
    with open(intent_file, "w", encoding="utf8") as file:
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


def get_user_input(rd_count):
    global inp
    try:
        if rd_count == 0:
            print("AIYU: 您好, 我是巨聪明AIYU, 有什么可以帮助您的吗？")
            inp = input("请开始和AIYU的对话: ")
            inp.replace("请开始和AIYU的对话: ", "")
        else:
            inp = input("您: ")
            inp.replace("您: ", "")
    except IOError:
        print("AIYU: 对不起，AIYU没听懂 T_T。")
    return inp


def get_current_time():
    curr_time = datetime.now()
    curr_time_format = curr_time.strftime("%H:%M:%S")
    return curr_time_format


def get_ai_username(mode):
    global username
    if mode == "dev":
        username = "AIYU: "
    elif mode == "discord":
        username = ""
    return username
