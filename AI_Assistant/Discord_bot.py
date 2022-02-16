import numpy as np
import discord
import Aiyu
import main
from Tools import support_fnc

# Creating Lists

discord_bot = Aiyu.Aiyu(main.training_set, main.output_data, main.words, main.labels, main.docs_x, main.docs_y,
                        main.model, main.intent_file, main.intents, main.tags, main.patterns, main.response_list,
                        main.language, main.init_state)
discord_bot.data_processor()
discord_bot.train_model(8, 8, 150)

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
            mention = "<@!{userid}>".format(userid=self.user.id)
            if mention in str(message.content):
                await message.channel.send(discord_bot.chat_dc(message).format(message))



client = chatbotAIYU()
# STARTING POINT
ai_token = ""
client.run(ai_token)
