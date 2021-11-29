import numpy as np
import tensorflow
import random
import json
import nltk
import numpy
import tflearn
import jieba
import stopwordsiso
import pickle
from nltk.stem.lancaster import \
    LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
from stopwordsiso import stopwords

stemmer = LancasterStemmer()
# Open the json file with the train data
with open("../Language_Data/Law_Data.json", encoding="utf8") as file:
    # returning the Json object
    data = json.load(file)
    # print(data)

# Comment this out after updating Law_Data json file
# try:
#     with open("data.pickle", "rb") as f:
#         words, labels, training, output = pickle.load(f)
# except:

words = []
labels = []
# Each pattern should be tagged with a law_type stored in docs_y
docs_x = []  # stores the tokenized word lists
docs_y = []  # stores the law_type for each tokenized word list

# Iterate through the "intents" in the json file.
for intent in data["law_database"]:
    # For each pattern inside each intent, we want to tokenize the words that are in sentences.
    for pattern in intent["law_keywords"]:
        tokenized_list_of_words = nltk.word_tokenize(pattern)  # This must be changed to the Jieba version of tokenizer.
        words.extend(tokenized_list_of_words)  # add the tokenized words into the "words" list
        docs_x.append(tokenized_list_of_words)
        docs_y.append(intent["law_type"])
        if intent["law_type"] not in labels:
            labels.append(intent["law_type"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]  # convert words into lower case
words = sorted(list(set(words)))  # sort and remove duplicates
labels = sorted(labels)

training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []
    wrds = [stemmer.stem(w) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

# Comment this out after the law_data file has been updated
# with open("data.pickle", "wb") as f:
#     pickle.dump((words, labels, training, output), f)

tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
# 2 hidden layers
# adding fully connected layer to the neural network which starts with the imput data and 8 neurons
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

# try:
#     model.load("model.tflearn")
# except:
model.fit(training, output, n_epoch=300, batch_size=19, show_metric=True)


#     model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def bag_of_words_chinese(s, words):
    # SETUP for Chinese sentence processing
    stop_words = set(stopwords(["zh"]))
    bag_chinese = [0 for _ in range(len(words))]
    s_words = jieba.cut(s)

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


def chat():
    print("请开始说话（输入 quit 来结束对话）")
    while True:
        inp = input("您： ")
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words_chinese(inp, words)])[0]
        print(results)
        results_index = numpy.argmax(results)
        law_type = labels[results_index]
        # probability threshold
        print("this is: " + law_type + "result accuracy: ", results_index)

        if results[results_index] > 0.7:
            for tg in data["law_database"]:
                if tg['law_type'] == law_type:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("对不起，Yu不知道您在说什么，能重复一下您的问题吗？")


chat()
