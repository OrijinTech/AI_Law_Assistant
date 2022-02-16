import sys
from AI_Assistant import Aiyu
from AI_Assistant import AI_StateMachine
import jieba

# Creating Lists
training_set = []
output_data = []
words = []
labels = []
docs_x = []
docs_y = []
model = None

# AI MODEL TRAINING DATA SETS:
intent_file = "Language_Data/Law_Data.json" # 普法数据
# intent_file = "Language_Data/intents.json" # 英文版社交机器人
# intent_file = "Language_Data/personalAI.json"  # 中文版社交机器人
# intent_file = "../Language_Data/personalAI.json"  # 中文版社交机器人(Discord私人版)
# intent_file = "../Language_Data/publicAI.json"  # 中文版社交机器人(Discord版)
# intent_file = "../Language_Data/Law_Data.json" # 普法AI(Discord版)

intents = "intents"
tags = "category"
patterns = "patterns"
response_list = "responses"
language = "ch"
init_state = AI_StateMachine.States.CHAT


def buildAI():
    aiyu = Aiyu.Aiyu(training_set, output_data, words, labels, docs_x, docs_y, model, intent_file, intents, tags,
                     patterns, response_list, language, init_state)
    aiyu.data_processor()
    aiyu.train_model(16, 8, 120)
    aiyu.chat()


def main(start_var):
    buildAI()
    return 0


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])
