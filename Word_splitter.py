# 这个是文件是用来做分词的


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import jieba  # 结巴分词 工具

def setup_jieba():
    jieba.initialize()

def fen_ci(message, cut_mode):
    seg_list = jieba.cut(message, cut_mode)
    return list(seg_list)
