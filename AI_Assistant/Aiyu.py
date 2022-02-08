import numpy as np
import numpy
import tflearn
from AI_Assistant import AI_StateMachine
from nltk.stem.lancaster import \
    LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
from Tools import support_fnc


class Aiyu:
    # Constructor
    def __init__(self, training_set, output_data, words, labels, docs_x, docs_y, ai_model,
                 intent_file, intents, tags, patterns, response_list, language, state):
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

    def train_model(self, num_neurons, batch_size, epoch_num):
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

    def pick_response(self, inp, results, results_index, conversation_type, labels):
        if results[results_index] > 0.7:  # probability threshold
            resp_list = []
            for tg in support_fnc.open_file(self.intent_file, "N")[self.intents]:
                if tg[self.tags] == conversation_type:
                    responses = tg[self.response_list]
                    resp_list.extend(responses)
                    if conversation_type == "时间":
                        print("AIYU: ", responses[support_fnc.get_max_similarity_percentage(inp, resp_list)], support_fnc.get_current_time())
                    elif conversation_type == "学习模式":
                        self.state = AI_StateMachine.States.LEARN
                        print("AIYU: ", responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
                    else:
                        print("AIYU: ", responses[support_fnc.get_max_similarity_percentage(inp, resp_list)])
        else:
            print("AIYU: 对不起，AIYU没听懂 T_T。")


    def chat(self):
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
                results = self.model.predict([support_fnc.bag_of_words(inp, self.words, self.language)])[0]  # Chinese Version
                results_index = numpy.argmax(results)
                conversation_type = self.labels[results_index]
                support_fnc.report_train_results(results, results_index, conversation_type, self.labels) # Report Results
                self.pick_response(inp, results, results_index, conversation_type, self.labels)
                round_count += 1
            # Learn State
            if self.state == AI_StateMachine.States.LEARN:
                while True:
                    learn_pattern = input("AIYU: 请输入您要我学习的文献：")
                    print("AIYU: 这是关于什么的对话？")
                    learn_type = input("您： ")
                    if learn_type in self.docs_y:
                        support_fnc.add_pattern(self.intent_file, self.intents, self.tags, self.patterns, learn_pattern, learn_type)
                    keep_learn = input("AIYU: 还有其他要我学习的吗？(Y/N)： ")
                    if keep_learn == "N":
                        self.state = AI_StateMachine.States.CHAT
                        round_count += 1
                        break
            # Quit the program
            if self.state == AI_StateMachine.States.QUIT:
                break

