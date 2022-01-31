# import numpy as np
# import tensorflow
# import random
# import json
# import nltk
# import numpy
# import tflearn
# import jieba
# import AI_StateMachine
# from nltk.stem.lancaster import \
#     LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
# from stopwordsiso import stopwords
# from Tools import data_modification
# from Tools import support_fnc
from difflib import SequenceMatcher
#
# # Global Variables
# initial_state = AI_StateMachine.States.CHAT
#
# stemmer = LancasterStemmer()
# # Open the json file with the train data
# with open("../Language_Data/Law_Data.json", encoding="utf8") as file:
#     # returning the Json object
#     data = json.load(file)
#
# # Comment this out after updating Law_Data json file
# # try:
# #     with open("data.pickle", "rb") as f:
# #         words, labels, training, output = pickle.load(f)
# # except:
#
# #Lists
# words = []
# labels = []
# docs_x = []  # stores the tokenized word lists
# docs_y = []  # stores the law_type for each tokenized word list. Each pattern should be tagged with a law_type stored in docs_y
#
# # Iterate through the "intents" in the json file.
# for intent in data["law_database"]:
#     # For each pattern inside each intent, we want to tokenize the words that are in sentences.
#     for pattern in intent["law_keywords"]:
#         tokenized_list_of_words = jieba.cut(pattern, cut_all=False)  # This is the tokenizer for the Chinese language.
#         toklist = list(tokenized_list_of_words)
#         words.extend(toklist)  # add the tokenized words into the "words" list
#         docs_x.append(toklist)
#         docs_y.append(intent["law_type"])
#         if intent["law_type"] not in labels:
#             labels.append(intent["law_type"])
#
# words = [stemmer.stem(w.lower()) for w in words if w not in "?"]  # convert words into lower case
# # words = sorted(list(set(words)))  # sort and remove duplicates
# labels = sorted(labels)
# training = []
# output = []
# out_empty = [0 for _ in range(len(labels))]
#
# for x, doc in enumerate(docs_x):
#     bag = []
#     wrds = [stemmer.stem(w) for w in doc]
#     for w in words:
#         if w in wrds:
#             bag.append(1)
#         else:
#             bag.append(0)
#     output_row = out_empty[:]
#     output_row[labels.index(docs_y[x])] = 1
#     training.append(bag)
#     output.append(output_row)
# training = np.array(training)
# output = np.array(output)
#
# # Comment this out after the law_data file has been updated
# # with open("data.pickle", "wb") as f:
# #     pickle.dump((words, labels, training, output), f)
# #
# # tensorflow.compat.v1.reset_default_graph()
#
# num_neurons = 12
# batchsize = 12
# epoch_num = 120
# # Input Layer
# net = tflearn.input_data(shape=[None, len(training[0])])
# # Hidden Layers
# net = tflearn.fully_connected(net, num_neurons)
# net = tflearn.fully_connected(net, num_neurons)
# net = tflearn.fully_connected(net, num_neurons)
# # Output Layer
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# # Model
# net = tflearn.regression(net)
# model = tflearn.DNN(net)
#
# model.fit(training, output, n_epoch=epoch_num, batch_size=batchsize,
#           show_metric=True)  # show_metric=True if you want training report
#
#
# # Comment out try catch, just leave model.fit for training
# # try:
# #     model.load("model.tflearn")
# # except:
# #     model.fit(training, output, n_epoch=epoch_num, batch_size=batchsize, show_metric=True)
# #     model.save("model.tflearn")
#
# def bag_of_words_chinese(s, words):
#     # SETUP for Chinese sentence processing
#     stop_words = set(stopwords(["zh"]))
#     bag_chinese = [0 for _ in range(len(words))]
#     s_words = jieba.cut(s, cut_all=False)
#
#     # stopwords filtration
#     s_words_filtered = []
#     for w in s_words:
#         if w not in stop_words:
#             s_words_filtered.append(w)
#     # unicode_s_words_filtered = [i.decode('utf-8') for i in s_words_filtered]
#     for se in s_words_filtered:
#         for i, w in enumerate(words):
#             if w == se:
#                 bag_chinese[i] = 1
#     return np.array(bag_chinese)
#
#
def check_similarity(a, b):
    return SequenceMatcher(None, a, b).quick_ratio()


def get_max_similarity_percentage(sentence, list_of_resp):
    responses_prob = []
    for sentences_in_list in list_of_resp:
        similarity_percentage = check_similarity(sentence, sentences_in_list)
        responses_prob.append(similarity_percentage)
    max_prob = max(responses_prob)
    return responses_prob.index(max_prob)
#
#
# def chat():
#     support_fnc.clear_chat()
#     global initial_state
#     while True:
#         # Chat State
#         if initial_state == AI_StateMachine.States.CHAT:
#             print("助手YU: 您好, 我是普法小助手YU, 有什么可以帮助您的吗？（输入 quit 来结束对话）")
#             while True:
#                 inp = input("您： ")
#                 if inp.lower() == "quit":
#                     initial_state = AI_StateMachine.States.QUIT
#                     break
#                 results = model.predict([bag_of_words_chinese(inp, words)])[0]  # Chinese Version
#                 #print(results)
#                 results_index = numpy.argmax(results)
#                 law_type = labels[results_index]
#                 result_percentage = round(results[results_index] * 100, 2)
#                 if results[results_index] > 0.7:
#                     print("YU有", result_percentage, "% 的确定性")
#                 else:
#                     print("很抱歉, 我只有", result_percentage, "% 的确定性")
#                 if results[results_index] > 0.7:  # probability threshold
#                     resp_list = []
#                     for tg in data["law_database"]:
#                         if tg['law_type'] == law_type:
#                             responses = tg['responses']
#                             resp_list.extend(responses)
#                             print("助手YU: ", responses[get_max_similarity_percentage(inp, resp_list)])
#                 else:
#                     print("助手YU: ", "对不起，Yu不知道您在说什么，如果想让Yu学习新东西的话请按”Y“. \n您也可以继续问答，继续问题请按”N“.")
#                     maint_input = input("请输入Y/N: ")
#                     if maint_input == "Y":
#                         initial_state = AI_StateMachine.States.LEARN
#                         break
#                 if law_type == "学习模式":
#                     initial_state = AI_StateMachine.States.LEARN
#                     break
#         # Learn State
#         if initial_state == AI_StateMachine.States.LEARN:
#             while True:
#                 learn_pattern = input("助手YU: 请输入您要我学习的文献：")
#                 print("助手YU: 这是哪种法律种类？")
#                 learn_type = input("您： ")
#                 if learn_type in docs_y:
#                     data_modification.add_pattern(learn_pattern, learn_type)
#                 keep_learn = input("助手YU: 还有其他要我学习的吗？(Y/N)： ")
#                 if keep_learn == "N":
#                     initial_state = AI_StateMachine.States.CHAT
#                     break
#         # Quit the program
#         if initial_state == AI_StateMachine.States.QUIT:
#             break
#
# #Starting Point
# chat()