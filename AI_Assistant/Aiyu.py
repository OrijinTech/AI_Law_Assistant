import keras
import numpy as np
import os
import numpy
import tflearn
import discord
import tensorflow
from keras.models import load_model
from keras.models import save_model
from AI_Assistant import AI_StateMachine
from nltk.stem.lancaster import \
    LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
from Tools import support_fnc
import h5py



# 用于创建种类识别object的框架。
class Aiyu:
    # Constructor
    def __init__(self, training_set, output_data, words, labels, docs_x, docs_y, ai_model,
                 intent_file, intents, tags, patterns, response_list, language, state, model_name):
        '''
        AI Constructor
        :param training_set: 训练数据（list）
        :param output_data: 输出数据（list）
        :param words: 词语容器（list）
        :param labels: 分类词容器（list）
        :param docs_x: 训练数据容器（list）
        :param docs_y: 训练数据容器（list)
        :param ai_model: 模型容器（None）
        :param intent_file: .json文件路径（str）
        :param intents: .json文件标题名称（str）
        :param tags: .json内分类词总称（str）
        :param patterns: .json关键词句总称（str）
        :param response_list: .json内回复词句总称（str）
        :param language:数据语言。en=英文，ch=中文 （str）
        :param state: State Machine初始 state（str）
        '''
        self.training = list(training_set)
        self.output = list(output_data)
        self.words = list(words)
        self.labels = list(labels)
        self.docs_x = list(docs_x)
        self.docs_y = list(docs_y)
        self.model = ai_model
        self.intent_file = intent_file
        self.intents = intents
        self.tags = tags
        self.patterns = patterns
        self.response_list = response_list
        self.language = language
        self.state = state
        self.model_name = model_name

    def data_processor(self):
        '''
        初步数据处理，提取.json文件内的数据，存入对应容器中。
        :return: None
        '''
        print("Processing data for model training.")
        data = support_fnc.open_file(self.intent_file, "Y")
        # Iterate through the "intents" in the json file.
        for intent in data[self.intents]:
            # For each pattern inside each intent, we want to tokenize the words that are in sentences.
            for pattern in intent[self.patterns]:
                tokenized_list_of_words = support_fnc.split_sentence(pattern, self.language)
                # add the tokenized words into the "words" list (only if the word list is not empty)
                if len(tokenized_list_of_words) > 0:
                    # print("list", tokenized_list_of_words, "added")
                    self.words.extend(tokenized_list_of_words)
                    self.docs_x.append(tokenized_list_of_words)
                self.docs_y.append(intent[self.tags])
                if intent[self.tags] not in self.labels:
                    self.labels.append(intent[self.tags])
        # prepare data for model training
        stemmer = LancasterStemmer()
        if self.language == "en":
            self.words = [stemmer.stem(w.lower()) for w in self.words if w not in "?"]  # convert words into lower case
        self.words = sorted(list(self.words))  # sort and remove duplicates
        self.labels = sorted(self.labels)
        labels = self.labels
        out_empty = [0 for _ in range(len(labels))]
        for x, doc in enumerate(self.docs_x):
            bag = []
            wrds = [stemmer.stem(w) for w in doc]
            for w in self.words:
                if w in wrds:
                    bag.append(1)
                else:
                    bag.append(0)
            output_row = out_empty[:]
            output_row[labels.index(self.docs_y[x])] = 1
            self.training.append(bag)
            self.output.append(output_row)
        # print(self.training)
        self.training = np.array(self.training)
        self.output = np.array(self.output)
        print("Finished processing data.")

    def train_model(self, num_neurons, batch_size, epoch_num):
        '''
        训练AI模型，更多在 http://tflearn.org/models/dnn/
        :param num_neurons: 神经元数量
        :param batch_size: 批量训练大小
        :param epoch_num: 迭代数量
        :return: None
        '''
        print("Training the best model.")
        training_data = self.training
        output_data = self.output
        # Input Layer
        net = tflearn.input_data(shape=[None, len(self.training[0])])
        # Hidden Layers
        # for layer in range(num_layer):
        net = tflearn.fully_connected(net, num_neurons)
        net = tflearn.fully_connected(net, num_neurons)
        # Output Layer
        net = tflearn.fully_connected(net, len(output_data[0]), activation="softmax")
        # Model
        net = tflearn.regression(net)
        model = tflearn.DNN(net)
        print("Starting fitting")
        model.fit(training_data, output_data, n_epoch=epoch_num, batch_size=batch_size,
                  show_metric=True)  # show_metric=True if you want training report
        self.model = model

    # model name = model.tflearn
    def save_model(self, path_name, model_name):
        '''
        保存模型（未完成）
        :param path_name: 模型储存位置（str）
        :param model_name: 模型名称（str）
        :return: None
        '''
        save_parameter = path_name + "/" + model_name
        save_model(self.model, 'model1.h5', save_format='h5')
        print("Model_Saved")

    def loading_model(self, path_name, model_name):
        '''
        加载AI模型
        :param path_name: 模型储存位置
        :param model_name: 模型名称（str）
        :return: None
        '''
        load_parameter = path_name + "/" + model_name
        self.model = load_model(load_parameter)

    def check_model(self, path_name, model_name):
        '''
        检查模型是否存在
        :param path_name: 模型储存位置
        :param model_name: 模型名称
        :return: Boolean
        '''
        check_parameter = path_name + "/" + model_name
        try:
            self.model.load(check_parameter)
            return True
        except:
            print("Model is not present, you need to train the model.")
            return False

    def prepare_model(self, num_neurons, batch_size, epoch_num, path_name="AI_Models", retrain_model="N"):
        if retrain_model == "Y":
            print("Retraining Model")
            self.train_model(num_neurons, batch_size, epoch_num)
            self.save_model(path_name, self.model_name)
        elif self.check_model(path_name, self.model_name):
            print("Loading Model")
            self.loading_model(path_name, self.model_name)
        else:
            print("Training New Model")
            self.train_model(num_neurons, batch_size, epoch_num)
            self.save_model(path_name, self.model_name)
        # else:
        #     self.loading_model(path_name, self.model_name)

    def pick_response(self, inp, results, results_index, conversation_type, labels, mode):
        '''
        选择输出文案
        :param inp: 用户输入（str）
        :param results: 模型预测的分类概率数据（list）
        :param results_index: 分类数据百分比清单的index（int）
        :param conversation_type: 分类名称（str）
        :param labels: 分类清单（list）
        :param mode: 用户使用模式dev=developer，discord=用户（str）
        :return: 输出给用户的文案（str）
        '''
        global response_return
        if results[results_index] > 0.7:  # probability threshold
            resp_list = []
            for tg in support_fnc.open_file(self.intent_file, "N")[self.intents]:
                if tg[self.tags] == conversation_type:
                    responses = tg[self.response_list]
                    resp_list.extend(responses)
                    if conversation_type == "时间":
                        response_return = str(support_fnc.get_ai_username(mode)) + str(
                            responses[support_fnc.get_max_similarity_percentage(inp, resp_list)]) + " " + str(
                            support_fnc.get_current_time())
                    elif conversation_type == "学习模式":
                        self.state = AI_StateMachine.States.LEARN
                        response_return = str(support_fnc.get_ai_username(mode)) + str(
                            responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
                    elif conversation_type == "爱好":
                        response_return = str(support_fnc.get_ai_username(mode)) + str(responses[
                                                                                           support_fnc.get_max_similarity_percentage(
                                                                                               inp,
                                                                                               resp_list)]) + "https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley"
                    else:
                        response_return = str(support_fnc.get_ai_username(mode)) + str(
                            responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
        else:
            response_return = str(support_fnc.get_ai_username(mode)) + "对不起，AIYU没听懂 T_T。"
        return response_return

    def chat(self):
        '''
        AI聊天功能
        :return: None
        '''
        self.state = AI_StateMachine.States.CHAT
        support_fnc.clear_chat()
        round_count = 0
        while True:
            # Chat State
            if self.state == AI_StateMachine.States.CHAT:
                inp = support_fnc.get_user_input(round_count)
                if inp.lower() == "quit":
                    self.state = AI_StateMachine.States.QUIT
                    break
                results = self.model.predict([support_fnc.bag_of_words(inp, self.words, self.language)])[
                    0]  # Chinese Version
                results_index = numpy.argmax(results)
                conversation_type = self.labels[results_index]
                support_fnc.report_train_results(results, results_index, conversation_type,
                                                 self.labels)  # Report Results
                print(self.pick_response(inp, results, results_index, conversation_type, self.labels, "dev"))
                round_count += 1
            # Learn State
            if self.state == AI_StateMachine.States.LEARN:
                while True:
                    learn_pattern = input("AIYU: 请输入您要我学习的文献：")
                    print("AIYU: 这是关于什么的对话？")
                    learn_type = input("您： ")
                    if learn_type in self.docs_y:
                        support_fnc.add_pattern(self.intent_file, learn_pattern,
                                                learn_type, self.intents, self.tags, self.patterns, )
                    keep_learn = input("AIYU: 还有其他要我学习的吗？(Y/N)： ")
                    if keep_learn == "N":
                        self.state = AI_StateMachine.States.CHAT
                        round_count += 1
                        break
            # Quit the program
            if self.state == AI_StateMachine.States.QUIT:
                break

    def chat_dc(self, message):
        '''
        聊天功能，discord版本
        :param message: discord得到的用户输入（message）
        :return: 输出给用户的文案（str）
        '''
        inp = message.content
        results = self.model.predict([support_fnc.bag_of_words(inp, self.words, self.language)])[0]
        results_index = numpy.argmax(results)
        conversation_type = self.labels[results_index]
        support_fnc.report_train_results(results, results_index, conversation_type, self.labels)
        return self.pick_response(inp, results, results_index, conversation_type, self.labels, "discord")
