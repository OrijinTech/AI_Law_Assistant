import sys
from AI_Assistant import AI_StateMachine, Aiyu
from Language_Data import file_names
import jieba
import tensorflow

# Bot Creation Parameters
intents = "intents"
category = "category"
patterns = "patterns"
response_list = "responses"
init_state = AI_StateMachine.States.CHAT
model = None
model_name = "LawType"


def build_chat_ai():
    aiyu = Aiyu.Aiyu(file_names.jsn_cbot_ch_n, intents, category, patterns, response_list, state=init_state,
                     ai_model=model, model_name=model_name)
    aiyu.data_processor("AIYU_PKL", force_process="Y", split_mode='Y')
    aiyu.construct_softmax_model(16, 8, 120, "AIYU_Core", retrain_model='Y')
    aiyu.chat()


def main(arg):
    build_chat_ai()
    return 0


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])
