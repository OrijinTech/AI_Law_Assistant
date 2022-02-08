import numpy as np
import discord
import Aiyu
import main
from Tools import support_fnc


# Creating Lists

discord_bot = Aiyu.Aiyu(main.training_set, main.output_data, main.words, main.labels, main.docs_x, main.docs_y, main.model, main.intent_file, main.intents, main.tags, main.patterns, main.response_list, main.language)
discord_bot.data_processor()
discord_bot.train_model(8, 8, 500)

global responses


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
            result = discord_bot.model.predict([support_fnc.bag_of_words(inp, discord_bot.words, discord_bot.language)])[0]
            result_index = np.argmax(result)
            tag = discord_bot.labels[result_index]
            if result[result_index] > 0.7:
                resp_list = []
                for tg in support_fnc.open_file(discord_bot.intent_file, "N")[discord_bot.intents]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                        resp_list.extend(responses)
                bot_respond = responses[support_fnc.get_max_similarity_percentage(inp, resp_list)]
                await message.channel.send(bot_respond.format(message))
            else:
                await message.channel.send("我听不太懂啊, 能再重复一下吗".format(message))


client = chatbotAIYU()
# STARTING POINT
ai_token = "OTM3MjI3NTUwNzIyMjQ4NzI0.YfYrLA.vTKwBSl7VKe952lP3ZRauVFq8oE"
client.run(ai_token)
