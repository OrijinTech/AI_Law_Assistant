import sys
from AI_Assistant import Aiyu
from AI_Assistant import AI_StateMachine
from Language_Data import file_names
import jieba
import tensorflow

# Bot Creation Parameters
global training_set
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



def build_chat_ai():
    aiyu = Aiyu.Aiyu(training_set, output_data, words, labels, docs_x, docs_y, file_names.jsn_cbot_ch_n, intents, category,
                     patterns, response_list, language, state=init_state, ai_model=model, model_name=model_name)
    aiyu.data_processor("AIYU_PKL", force_process="N", split_mode='Y')
    aiyu.construct_model(16, 8, 120, "AIYU_Core", retrain_model='N')
    aiyu.chat()


def main(start_var):
    build_chat_ai()
    return 0


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])

