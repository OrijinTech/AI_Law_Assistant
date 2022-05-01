import discord
import Aiyu
from AI_Assistant import AI_StateMachine
from Language_Data import file_names


# Bot Creation Parameters
training_set = []
output_data = []
words = []
labels = []
docs_x = []
docs_y = []
intents = "intents"
category = "category"
patterns = "patterns"
response_list = "responses"
language = "ch"
init_state = AI_StateMachine.States.CHAT
model = None
model_name = "LawType"


# Bot Creation
discord_bot = Aiyu.Aiyu(training_set, output_data, words, labels, docs_x, docs_y, file_names.jsn_cbot_ch_n, intents, category,
                 patterns, response_list, language, state=init_state, ai_model=model, model_name=model_name)
discord_bot.data_processor("AIYU_PKL", force_process="N", split_mode='Y')
discord_bot.construct_model(16, 8, 120, "AIYU_Core", retrain_model='N')

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
