import sys
from AI_Assistant import AI_StateMachine, Aiyu
from Language_Data import file_names
import jieba
import tensorflow




def build_chat_ai():
    aiyu = Aiyu.Aiyu(training_set, output_data, words, labels, docs_x, docs_y, file_names.jsn_cbot_ch_n, intents, category,
                     patterns, response_list, language, state=init_state, ai_model=model, model_name=model_name)
    aiyu.data_processor("AIYU_PKL", force_process="Y", split_mode='Y')
    aiyu.construct_model(16, 8, 120, "AIYU_Core", retrain_model='Y')
