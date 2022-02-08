import jieba
import nltk
import numpy as np
import numpy
import tflearn
from AI_Assistant import AI_StateMachine
from nltk.stem.lancaster import \
    LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
from Tools import support_fnc


class Aiyu:

    def __init__(self, training_set, output_data, words, labels, docs_x, docs_y, ai_model,
                 intent_file, intents, tags, patterns, language):
        self.training = list(training_set)
        self.output = list(output_data)
        self.words = list(words)
        self.labels = list(labels)
        self.docs_x = list(docs_x)
        self.docs_y = list(docs_y)
        self.model = ai_model
        self.intent_file = intent_file
        self.intents = intents  # str
        self.tags = tags  # str
        self.patterns = patterns  # str
        self.language = language

    def data_processor(self):
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

    def train_model(self, num_neurons, batchsize, epoch_num):
        print("Training the best model.")
        training_data = self.training
        output_data = self.output
        # Input Layer
        net = tflearn.input_data(shape=[None, len(self.training[0])])
        # Hidden Layers
        #for layer in range(num_layer):
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
                print("AIYU: 您好, 我是普法小助手AIYU, 有什么可以帮助您的吗？（输入 quit 来结束对话）")
                while True:
                    print("您: ")
                    inp = input()
                    if len(inp) > 0:
                        if inp.lower() == "quit":
                            initial_state = AI_StateMachine.States.QUIT
                            break
                        results = self.model.predict([support_fnc.bag_of_words_ch(inp, self.words)])[0]  # Chinese Version
                        results_index = numpy.argmax(results)
                        law_type = self.labels[results_index]
                        if results[results_index] > 0.7:  # probability threshold
                            resp_list = []
                            for tg in support_fnc.open_file(self.intent_file, "N")[self.intents]:
                                if tg['law_type'] == law_type:
                                    responses = tg['responses']
                                    resp_list.extend(responses)
                                    print("AIYU: ", responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
                        else:
                            print("AIYU: ", "对不起，AIYU不知道您在说什么，如果想让Yu学习新东西的话请按”Y“. \n您也可以继续问答，继续问题请按”N“.")
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
                    learn_pattern = input("AIYU: 请输入您要我学习的文献：")
                    print("AIYU: 这是哪种法律种类？")
                    learn_type = input("您： ")
                    if learn_type in self.docs_y:
                        support_fnc.add_pattern(learn_pattern, learn_type)
                    keep_learn = input("AIYU: 还有其他要我学习的吗？(Y/N)： ")
                    if keep_learn == "N":
                        initial_state = AI_StateMachine.States.CHAT
                        break
            # Quit the program
            if initial_state == AI_StateMachine.States.QUIT:
                break

