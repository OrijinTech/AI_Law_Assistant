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
    '''
    打开一个指定的.json文件
    :param file_name: 文件名和路径
    :param report: 显示是否正在打开文件。
    :return: .json文件里的数据，type any
    '''
    if report == "Y":
        print("Opening file: ", file_name)
    # Open the json file with the train data
    with open(file_name, encoding="utf8") as file:
        # returning the Json object
        data = json.load(file)
    return data


def clear_chat():
    '''
    清空当前输出屏
    :return: None
    '''
    for i in range(50):
        print("\n")


def check_similarity(a, b):
    '''
    检查两个str的相同度
    :param a: Str A
    :param b: Str B
    :return: 输出相似度百分比(%)
    '''
    return SequenceMatcher(None, a, b).quick_ratio()


def get_max_similarity_percentage(sentence, list_of_resp):
    '''
    对比输入文本与list里的所有句子，并给出list中最高相似度的句子。
    :param sentence: 输入的文本（sentence）
    :param list_of_resp: 文本的list（list of sentences）
    :return: 最大的相似度百分比（%）
    '''
    responses_prob = []
    for sentences_in_list in list_of_resp:
        similarity_percentage = check_similarity(word_filter_rem(sentence, "您好"), sentences_in_list)
        responses_prob.append(similarity_percentage)
    max_prob = max(responses_prob)
    return responses_prob.index(max_prob)


def bag_of_words(s, words, language):
    '''
    :param s: 句子
    :param words: 单词的list
    :param language: 分词语言 en=英文，ch=中文
    :return: numpy array
    '''
    bag_chinese = [0 for _ in range(len(words))]
    s_words = split_sentence(s, language)
    words_list = list(s_words)
    for se in words_list:
        for i, w in enumerate(words):
            if w == se:
                bag_chinese[i] = 1
    return np.array(bag_chinese)


def word_filter_rem(sentence, word):
    '''
    过滤一个句子内的指定词语
    :param sentence: 指定句子 (str)
    :param word: 指定被过滤词语 (str)
    :return: 新句子（str)
    '''
    word_list = split_sentence(sentence, "ch")
    updated_sent = ""
    for w in word_list:
        if w == word:
            word_list.remove(w)
    for st in word_list:
        updated_sent += st
    return updated_sent


def report_train_results(results, results_index, intent_type, labels):
    '''
    输出报告信息
    :param results: 所有分类的结果百分比（list）
    :param results_index: result list里最大的百分比的项目的指标/index（int）
    :param intent_type: 分类名称（str）
    :param labels: 所有分类（list）
    :return: None
    '''
    result_percentage = round(results[results_index] * 100, 2)
    print("---------REPORTING RESULTS---------"
          "\nProbabilities of Types:", "\n", results,
          "\nCategories:", "\n", labels,
          "\nIndex of the Type with Largest Probability:", results_index,
          "\nIndex of the Resulted Type:", intent_type,
          "\nProbability of the Type:", result_percentage, "%")


def split_sentence(sentence, language):
    '''
    分词工具
    :param sentence: 分词的原句（str）
    :param language: 句子语言（str）
    :return: 分词后生成一个词语清单（list）
    '''
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


def add_pattern(intent_file, pattern_to_learn, add_to_category, intents="intents", pattern="patterns", category="category"):
    '''
    添加新关键词句到.json文件中
    :param intent_file: .json文件名路径（str）
    :param intents: 文件开头名字（str）
    :param category: 所有分类分类名称（data）
    :param pattern: 所有分类关键词句（data）
    :param pattern_to_learn: 要加入的关键词句（str）
    :param add_to_category: 要加入关键词句的目标分类（str）
    :return: None
    '''
    with open(intent_file, "r", encoding="utf8") as file:
        # returning the Json object
        data = json.load(file)
        for intent in data[intents]:
            if intent[category] == add_to_category:
                intent[pattern].append(pattern_to_learn)
    with open(intent_file, "w", encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def add_category(intent_file, category):
    '''
    在.json文件内添加新的种类(category)
    :param intent_file: 目标.json文件
    :type intent_file: str
    :param category: 待添加的新种类
    :type category: str
    :return: None
    :rtype: None
    '''
    with open(intent_file, "r", encoding="utf8") as file:
        data = json.load(file)
        category_to_add = {"category": category, "patterns": [], "responses": [], "context_set": ""}
        data["intents"].append(category_to_add)
    with open(intent_file, "w", encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def update_json(intent_file, intents="intents", pattern="patterns", category="category"):
    '''
    更新.json文件
    :param intent_file: 目标.json文件
    :type intent_file: str
    :param intents: .json文件内对应的 "intents” 名称标题
    :type intents: str
    :param pattern: .json文件内对应的 "patterns” 名称标题
    :type pattern: str
    :param category: .json文件内对应的 "category” 名称标题
    :type category: str
    :return: None
    :rtype: None
    '''
    while True:
        create_bool = input("创建Category = C？ | 添加Pattern = P")
        if create_bool == "C":
            input_category = input("请输入创建的category：")
            cat_list = input_category.split("，")
            # adding all categories
            for cat in cat_list:
                add_category(intent_file, cat)
        elif create_bool == "P":
            learn_pattern = input("请输入添加的pattern：")
            add_to_category = input("请输入目标category")
            pat_list = learn_pattern.split("，")
            try:
                for pat in pat_list:
                    add_pattern(intent_file, pattern, pat, add_to_category, intents, category)
            except KeyError:
                print("找不到目标Category，请重试。")
        keep_learn = input("AIYU: 还有其他要我学习的吗？(Y/N)： ")
        if keep_learn == "N":
            break


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
    '''
    获取用户输入
    :param rd_count: 循环数（int）
    :return: 用户输入（str）
    '''
    global inp
    try:
        if rd_count == 0:
            print("AIYU: 您好, 我是AIYU, 有什么可以帮助您的吗？")
            inp = input("请开始和AIYU的对话: ")
            inp.replace("请开始和AIYU的对话: ", "")
        else:
            inp = input("您: ")
            inp.replace("您: ", "")
    except IOError:
        print("AIYU: 对不起，AIYU没听懂 T_T。")
    return inp


def get_current_time():
    '''
    获取现在时间
    :return: 时间（str）
    '''
    curr_time = datetime.now()
    curr_time_format = curr_time.strftime("%H:%M:%S")
    return curr_time_format


def get_ai_username(mode):
    '''
    获取AI用户名
    :param mode: 使用模式，现在有dev= developer，discord= 用户（str）
    :return: 用户名（str）
    '''
    global username
    if mode == "dev":
        username = "AIYU: "
    elif mode == "discord":
        username = ""
    return username
