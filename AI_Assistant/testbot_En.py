import numpy as np
import discord
import json
import nltk
import tflearn
import AI_StateMachine
from nltk.stem.lancaster import \
    LancasterStemmer  # Used to analyze words from the sentence (getting the root of the word --> only for English)
from Tools import support_fnc

# Global Variables
initial_state = AI_StateMachine.States.CHAT

stemmer = LancasterStemmer()
# Open the json file with the train data
with open("../Language_Data/intents.json", encoding="utf8") as file:
    # returning the Json object
    data = json.load(file)

words = []
labels = []
# Each pattern should be tagged with a law_type stored in docs_y
docs_x = []  # stores the tokenized word lists
docs_y = []  # stores the law_type for each tokenized word list
word_docs_x = []
# Iterate through the "intents" in the json file.
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokenized_list_of_words = nltk.word_tokenize(pattern)
        words.extend(tokenized_list_of_words)
        docs_x.append(tokenized_list_of_words)
        print(tokenized_list_of_words)
        docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]  # convert words into lower case
# words = sorted(list(set(words)))  # sort and remove duplicates
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

num_neurons = 8
batchsize = 8
epoch_num = 300
# Input Layer
net = tflearn.input_data(shape=[None, len(training[0])])
# Hidden Layers
net = tflearn.fully_connected(net, num_neurons)
net = tflearn.fully_connected(net, num_neurons)
# Output Layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# Model
net = tflearn.regression(net)
model = tflearn.DNN(net)

model.fit(training, output, n_epoch=epoch_num, batch_size=batchsize,
          show_metric=True)  # show_metric=True if you want training report


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


# def chat_en(): //Chat Bot EN test
#     global initial_state
#
#     while True:
#         # Chat State
#         if initial_state == AI_StateMachine.States.CHAT:
#             print("请开始说话（输入 quit 来结束对话）")
#             while True:
#                 inp = input("您： ")
#                 if inp.lower() == "quit":
#                     initial_state = AI_StateMachine.States.QUIT
#                     break
#                 results = model.predict([bag_of_words(inp, words)])[0]
#                 results_index = numpy.argmax(results)
#                 law_type = labels[results_index]
#                 result_percentage = round(results[results_index] * 100, 2)
#                 if results[results_index] > 0.7:
#                     print("YU有", result_percentage, "% 的确定性")
#                 else:
#                     print("很抱歉, 我只有", result_percentage, "% 的确定性")
#                 if results[results_index] > 0.7:  # probability threshold
#                     for tg in data["law_database"]:
#                         if tg['law_type'] == law_type:
#                             responses = tg['responses']
#                             print(random.choice(responses))
#                 else:
#                     print("对不起，Yu不知道您在说什么，如果想让Yu学习新东西的话请按”Y“. \n您也可以继续问答，继续问题请按”N“.")
#                     maint_input = input("请输入Y/N: ")
#                     if maint_input == "Y":
#                         initial_state = AI_StateMachine.States.LEARN
#                         break
#                 if law_type == "学习模式":
#                     initial_state = AI_StateMachine.States.LEARN
#                     break
#             # Learn State
#             if initial_state == AI_StateMachine.States.LEARN:
#                 while True:
#                     learn_pattern = input("请输入您要我学习的文献：")
#                     print("这是哪种法律种类？")
#                     learn_type = input("您： ")
#                     if learn_type in docs_y:
#                         data_modification.add_pattern(learn_pattern, learn_type)
#                     keep_learn = input("还有其他要我学习的吗？(Y/N)： ")
#                     if keep_learn == "N":
#                         initial_state = AI_StateMachine.States.CHAT
#                         break
#             # Quit the program
#             if initial_state == AI_StateMachine.States.QUIT:
#                 break

# client = discord.Client()
# @client.event
# async def messaging(self, message):
#     print('Logged in as')
#     print(self.user.name)
#     print(self.user.id)
#     print('------')
#     if message.author == client.user:
#         return
#     else:
#         inp = message.content
#         results = model.predict([bag_of_words(inp, words)])[0]
#         results_index = np.argmax(results)
#         law_type = labels[results_index]
#         if results[results_index] > 0.7:  # probability threshold
#             resp_list = []
#             for tg in data["intents"]:
#                 if tg['tag'] == law_type:
#                     responses = tg['responses']
#                     resp_list.extend(responses)
#                     bot_respond = responses[AI_bot.get_max_similarity_percentage(inp, resp_list)]
#                     await message.channel.send(bot_respond.format(message))
#         else:
#             await message.channel.send("对不起，AIYU不知道您在说什么Q_Q".format(message))


global responses
# DISCORD IMPLEMENTATION
class chatbotAIYU(discord.Client):

    async def on_ready(self):
        print('Logged in as')
        print(self.user.name)
        print(self.user.id)
        print('------')

    async def on_message(self, message):
        # we do not want the bot to reply to itself

        global responses
        if message.author.id == self.user.id:
            return

        else:
            inp = message.content
            result = model.predict([bag_of_words(inp, words)])[0]
            result_index = np.argmax(result)
            tag = labels[result_index]
            print(result[result_index])
            if result[result_index] > 0.7:
                resp_list = []
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                        resp_list.extend(responses)
                bot_respond = responses[support_fnc.get_max_similarity_percentage(inp, resp_list)]
                await message.channel.send(bot_respond.format(message))
            else:
                await message.channel.send("I didnt get that. Can you explain or try again.".format(message))


client = chatbotAIYU()
# STARTING POINT
client.run('OTM3MjI3NTUwNzIyMjQ4NzI0.YfYrLA.HQn-_mDuyBiVhDBUaVgv3ug9hrY')
