import jieba
import numpy as np
import numpy
import tflearn
from AI_Assistant import AI_StateMachine
from nltk.stem.lancaster import \
    LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
from Tools import data_modification
from Tools import support_fnc


class Aiyu:

    # Law AI Config:
    # File name: ../Language_Data/Law_Data.json"
    # intents: "law_database"
    # tag: "law_type"
    # patterns: "law_keywords"

    def __init__(self, training_set, output_data, words, labels, docs_x, docs_y, ai_model,
                 intent_file, intents, tags, patterns):
        self.training = list(training_set)
        self.output = list(output_data)
        self.words = list(words)
        self.labels = list(labels)
        self.docs_x = list(docs_x)
        self.docs_y = list(docs_y)
        self.model = ai_model
        self.intent_file = intent_file
        self.intents = str(intents)  # str
        self.tags = str(tags)  # str
        self.patterns = str(patterns)  # str

    def data_processor(self):
        print("Processing data for model training.")
        # Iterate through the "intents" in the json file.
        for intent in support_fnc.open_file(self.intent_file)[self.intents]:
            # For each pattern inside each intent, we want to tokenize the words that are in sentences.
            for pattern in intent[self.patterns]:
                # This is the tokenizer for the Chinese language.
                tokenized_list_of_words = jieba.cut(pattern, cut_all=False)
                toklist = list(tokenized_list_of_words)
                self.words.extend(toklist)  # add the tokenized words into the "words" list
                self.docs_x.append(toklist)
                self.docs_y.append(intent[self.tags])
                if intent[self.tags] not in self.labels:
                    self.labels.append(intent[self.tags])
        # prepare data for model training
        print("Start stemming.")
        stemmer = LancasterStemmer()
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in "?"]  # convert words into lower case
        # words = sorted(list(set(words)))  # sort and remove duplicates
        print("Sorting")
        labels = sorted(self.labels)
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
        self.training = np.array(self.training)
        self.output = np.array(self.output)
        print("Finished processing data.")

    # Law AI Config:
    # num_neurons = 12
    # batchsize = 12
    # epoch_num = 120
    def train_model(self, num_neurons, batchsize, epoch_num):
        print("Training the best model.")
        training_data = self.training
        output_data = self.output
        # Input Layer
        net = tflearn.input_data(shape=[None, len(self.training[0])])
        # Hidden Layers
        net = tflearn.fully_connected(net, num_neurons)
        net = tflearn.fully_connected(net, num_neurons)
        net = tflearn.fully_connected(net, num_neurons)
        # Output Layer
        net = tflearn.fully_connected(net, len(output_data[0]), activation="softmax")
        # Model
        net = tflearn.regression(net)
        model = tflearn.DNN(net)
        print("Starting fitting")
        model.fit(training_data, output_data, n_epoch=epoch_num, batch_size=batchsize,
                  show_metric=True)  # show_metric=True if you want training report
        self.model = model

    def chat(self):
        initial_state = AI_StateMachine.States.CHAT
        support_fnc.clear_chat()
        while True:
            # Chat State
            if initial_state == AI_StateMachine.States.CHAT:
                print("助手AIYU: 您好, 我是普法小助手AIYU, 有什么可以帮助您的吗？（输入 quit 来结束对话）")
                while True:
                    inp = input("您： ")
                    if inp.lower() == "quit":
                        initial_state = AI_StateMachine.States.QUIT
                        break
                    results = self.model.predict([support_fnc.bag_of_words_chinese(inp, self.words)])[
                        0]  # Chinese Version
                    # print(results)
                    results_index = numpy.argmax(results)
                    law_type = self.labels[results_index]
                    result_percentage = round(results[results_index] * 100, 2)
                    if results[results_index] > 0.7:
                        print("AIYU有", result_percentage, "% 的确定性,这是", law_type, "。")
                    else:
                        print("很抱歉, 我只有", result_percentage, "% 的确定性")
                    if results[results_index] > 0.7:  # probability threshold
                        resp_list = []
                        for tg in support_fnc.open_file(self.intent_file)[self.intents]:
                            if tg['law_type'] == law_type:
                                responses = tg['responses']
                                resp_list.extend(responses)
                                print("助手YU: ", responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
                    else:
                        print("助手YU: ", "对不起，AIYU不知道您在说什么，如果想让Yu学习新东西的话请按”Y“. \n您也可以继续问答，继续问题请按”N“.")
                        maint_input = input("请输入Y/N: ")
                        if maint_input == "Y":
                            initial_state = AI_StateMachine.States.LEARN
                            break
                    if law_type == "学习模式":
                        initial_state = AI_StateMachine.States.LEARN
                        break
            # Learn State
            if initial_state == AI_StateMachine.States.LEARN:
                while True:
                    learn_pattern = input("助手YU: 请输入您要我学习的文献：")
                    print("助手YU: 这是哪种法律种类？")
                    learn_type = input("您： ")
                    if learn_type in self.docs_y:
                        data_modification.add_pattern(learn_pattern, learn_type)
                    keep_learn = input("助手YU: 还有其他要我学习的吗？(Y/N)： ")
                    if keep_learn == "N":
                        initial_state = AI_StateMachine.States.CHAT
                        break
            # Quit the program
            if initial_state == AI_StateMachine.States.QUIT:
                break
