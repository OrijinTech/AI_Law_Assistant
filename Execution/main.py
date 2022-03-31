import sys
from AI_Assistant import Aiyu
from AI_Assistant import AI_StateMachine
import jieba
import tensorflow

# Creating Lists
training_set = []
output_data = []
words = []
labels = []
docs_x = []
docs_y = []
model = None
modelname = "LawType"

# AI MODEL TRAINING DATA SETS:
# intent_file = "../Language_Data/Law_Datasets/Law_Data.json"   # 普法AI
# intent_file = "Language_Data/intents.json"                    # 英文版社交机器人
intent_file = "../Language_Data/Other_Data/personalAI.json"   # 社交机器人私人版
# intent_file = "../Language_Data/Other_Data/publicAI.json"     # 社交机器人普通版

intents = "intents"
category = "category"
patterns = "patterns"
response_list = "responses"
language = "ch"
init_state = AI_StateMachine.States.CHAT


def buildAI():
    aiyu = Aiyu.Aiyu(training_set, output_data, words, labels, docs_x, docs_y, model, intent_file, intents, category,
                     patterns, response_list, language, init_state, modelname)
    aiyu.data_processor()
    aiyu.prepare_model(16, 8, 120, "AIYU_Core", retrain_model="Y")
    aiyu.chat()


def main(start_var):
    buildAI()
    return 0


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])
