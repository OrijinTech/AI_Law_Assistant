import numpy as np
import discord
import Aiyu
import main
from Tools import support_fnc


# Creating Lists

discord_bot = Aiyu.Aiyu(main.training_set, main.output_data, main.words, main.labels, main.docs_x, main.docs_y, main.model, main.intent_file, main.intents, main.tags, main.patterns, main.language)
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
            result = discord_bot.model.predict([support_fnc.bag_of_words_en(inp, discord_bot.words)])[0]
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
                await message.channel.send("I didnt get that. Can you explain or try again.".format(message))


client = chatbotAIYU()
# STARTING POINT
ai_token = ""
client.run(ai_token)
