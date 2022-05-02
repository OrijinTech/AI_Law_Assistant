import numpy as np
import numpy
import tensorflow as tf
import os
import random
import h5py
import keras
from tensorflow import keras
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.models import load_model
from keras.models import save_model
from AI_Assistant.AI_StateMachine import States
from nltk.stem.lancaster import LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
from Tools import support_fnc


# 用于创建种类识别object的框架。
class Aiyu:
    # Constructor
    def __init__(self, intent_file, intents, tags, patterns, response_list, training_set='', output_data='',
                 words='', labels='', docs_x='', docs_y='', language='ch', state=States.CHAT, ai_model=None,
                 model_name=''):
        """
        :param intent_file: .json文件路径（str）
        :param intents: .json文件标题名称（str）
        :param tags: .json内分类词总称（str）
        :param patterns: .json关键词句总称（str）
        :param response_list: .json内回复词句总称（str）
        :param training_set: 训练数据（list）
        :param output_data: 输出数据（list）
        :param words: 词语容器（list）
        :param labels: 分类词容器（list）
        :param docs_x: 输入训练数据（list）
        :param docs_y: 输出训练数据（list)
        :param language: 数据语言。en=英文，ch=中文 （str）
        :param state: State Machine初始 state（str）
        :param ai_model: AI 模型文件名
        :param model_name: AI 模型名
        """
        # 创建AI模型信息
        self.training = list(training_set)
        self.output = list(output_data)
        self.words = list(words)
        self.labels = list(labels)
        self.docs_x = list(docs_x)
        self.docs_y = list(docs_y)
        self.intent_file = intent_file
        self.intents = intents
        self.tags = tags
        self.patterns = patterns
        self.response_list = response_list
        self.language = language
        self.state = state
        self.model = ai_model
        self.model_name = model_name

    def data_processor(self, pickle_file, force_process="N", split_mode="Y"):
        """
        初步数据处理，提取.json文件内的数据，存入对应容器中。
        :return: None
        """
        if not support_fnc.has_pickle(pickle_file) or force_process == "Y":
            print("Processing data for model training.")
            data = support_fnc.open_file(self.intent_file, report="Y")
            # Iterate through the "intents" in the json file.
            for intent in data[self.intents]:
                # For each pattern inside each intent, we want to tokenize the words that are in sentences.
                for pattern in intent[self.patterns]:
                    tokenized_list_of_words = support_fnc.split_sentence(pattern, split_type=split_mode, language=self.language)
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
            self.words = sorted(list(set(self.words)))  # sort and remove duplicates
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
            support_fnc.save_pickle(pickle_file, self.training, self.output, self.words, self.labels, self.docs_x, self.docs_y)
            print("Finished processing data.")
        else:
            self.training, self.output, self.words, self.labels, self.docs_x, self.docs_y = support_fnc.load_pickle(pickle_file)


# ==================================================================================================================================== #
#                                                         MODEL DESIGN
# ==================================================================================================================================== #


    def construct_softmax_model(self, num_neurons, batch_size, epoch_num, model_name, retrain_model='N', path_name="../AI_Models"):
        """
        训练AI模型，更多在 https://tflearn.org/models/dnn/
        :param path_name: 模型储存路径
        :param folder_name: 模型储存文件夹名
        :param model_name: 模型储存文件前缀名
        :param retrain_model:是否重新训练模型
        :param num_neurons: 神经元数量
        :param batch_size: 批量训练大小
        :param epoch_num: 迭代数量
        :return: None
        """
        print("Training the best model.")
        # Preparing Variables
        training_data = self.training
        output_data = self.output
        # Input Layer
        inp_layer = Input(shape=(len(training_data[0]), ))  # example: input shape = (None, 689)
        # Hidden Layers
        hidden1 = Dense(units=num_neurons, activation="tanh")(inp_layer)
        hidden2 = Dense(units=num_neurons, activation="tanh")(hidden1)
        # hidden3 = Dense(num_neurons)(hidden2)
        # Output Layers
        output = Dense(len(output_data[0]), activation='softmax')(hidden2)
        model = Model(inputs=inp_layer, outputs=output)
        # Compile Model
        model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["categorical_crossentropy"])
        self.train_model(model, batch_size, epoch_num, model_name, retrain_model, path_name)


    # def construct_relu_model(self, num_neurons, batch_size, epoch_num, model_name, retrain_model="N", path_name="../AI_Models"):



    def train_model(self, model, batch_size, epoch_num, model_name, retrain_model='N', path_name="../AI_Models"):
        save_parameter = path_name + "/" + model_name
        # Training Decision
        if retrain_model == "Y":  # If the user wants to retrain the model.
            print("Retraining Model")
            # Train model
            model.fit(self.training, self.output, epochs=epoch_num, batch_size=batch_size)
            # Model data
            model_data = model.evaluate(self.training, self.output, batch_size=batch_size)
            print(model_data)
            # Save Model
            model.save(save_parameter)
            self.model = model
        elif os.path.isdir(save_parameter):  # If the model folder exists
            print("Model detected.")
            # Loading Model
            model = tf.keras.models.load_model(save_parameter)
            self.model = model
        else:
            print("Model not detected.")
            # Train model
            model.fit(self.training, self.output, epochs=epoch_num, batch_size=batch_size)
            # Model data
            model_data = model.evaluate(self.training, self.output, batch_size=batch_size)
            print(model_data)
            # Save Model
            print("Saving Model")
            model.save(save_parameter)
            self.model = model


# ==================================================================================================================================== #
#                                                         Normal Chat Functions
# ==================================================================================================================================== #


    def model_predict(self, inp):
        """
        使用当前的AI来预测种类
        :param inp: 输入字符（str）
        :return: 所有种类的预测概率（list of double）
        """
        results = self.model.predict(tf.expand_dims(support_fnc.bag_of_words(inp, self.words, self.language), axis=0))[0]
        # List of doubles of percentages of each category.
        return results


    def pick_response(self, inp, results, results_index, conversation_type, labels, mode, pick_mode='rand'):
        """
        选择输出文案
        :param pick_mode:
        :param inp: 用户输入（str）
        :param results: 模型预测的分类概率数据（list）
        :param results_index: 分类数据百分比清单的index（int）
        :param conversation_type: 分类名称（str）
        :param labels: 分类清单（list）
        :param mode: 用户使用模式dev=developer，discord=用户（str）
        :param pick_mode: 选择输出模式, rand = 随机选择输出（str）
        :return: 输出给用户的文案（str）
        """
        global response_return
        if results[results_index] > 0.7:  # probability threshold
            resp_list = []
            for tg in support_fnc.open_file(self.intent_file, report="N")[self.intents]:
                if tg[self.tags] == conversation_type:
                    responses = tg[self.response_list]
                    resp_list.extend(responses)
                    if conversation_type == "时间":
                        response_return = str(support_fnc.get_ai_username(mode)) + str(
                            responses[support_fnc.get_max_similarity_percentage(inp, resp_list)]) + " " + str(
                            support_fnc.get_current_time())
                    elif conversation_type == "学习模式":
                        self.state = States.LEARN
                        response_return = str(support_fnc.get_ai_username(mode)) + str(
                            responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
                    elif conversation_type == "爱好":
                        response_return = str(support_fnc.get_ai_username(mode)) + str(responses[
                                                                                           support_fnc.get_max_similarity_percentage(
                                                                                               inp,
                                                                                               resp_list)]) + "https://www.youtube.com/watch?v=dQw4w9WgXcQ&ab_channel=RickAstley"
                    else:
                        try:
                            if pick_mode == 'rand':
                                response_return = str(support_fnc.get_ai_username(mode)) + str(random.choice(responses))
                            else:
                                response_return = str(support_fnc.get_ai_username(mode)) + str(
                                    responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
                        except TypeError:
                            print("WARNING: I could not find any answers for you. Please check the json file.")
        else:
            response_return = str(support_fnc.get_ai_username(mode)) + "对不起，AIYU没听懂 T_T。"
        return response_return

    def chat(self):
        """
        AI聊天功能
        :return: None
        """
        self.state = States.CHAT
        support_fnc.clear_chat()
        round_count = 0
        while True:
            # Chat State
            if self.state == States.CHAT:
                inp = support_fnc.get_user_input(round_count)
                if inp.lower() == "quit":
                    self.state = States.QUIT
                    break
                # Prediction
                results = self.model_predict(inp)  # axis = 0 adjusts the dimension for input shape
                print(results)
                results_index = numpy.argmax(results)
                conversation_type = self.labels[results_index]
                # support_fnc.report_train_results(results, results_index, conversation_type, self.labels)  # Report Results
                print(self.pick_response(inp, results, results_index, conversation_type, self.labels, "dev"))
                round_count += 1
            # Learn State
            if self.state == States.LEARN:
                support_fnc.update_json(self.intent_file, self.intents, self.patterns, self.tags)
                self.state = States.CHAT
            # Quit the program
            if self.state == States.QUIT:
                break


# ==================================================================================================================================== #
#                                                         Discord Chat Functions
# ==================================================================================================================================== #


    def chat_dc(self, message):
        """
        聊天功能，discord版本
        :param message: discord得到的用户输入（message）
        :return: 输出给用户的文案（str）
        """
        inp = message.content
        results = self.model.predict(tf.expand_dims(support_fnc.bag_of_words(inp, self.words, self.language), axis=0))[0]
        results_index = numpy.argmax(results)
        conversation_type = self.labels[results_index]
        # support_fnc.report_train_results(results, results_index, conversation_type, self.labels)
        return self.pick_response(inp, results, results_index, conversation_type, self.labels, "discord")

