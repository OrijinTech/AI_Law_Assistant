# 这个是文件是用来做分词的


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import jieba  # 结巴分词 工具

# 初始化结巴分词
def setup_jieba():
    '''
    初始化分词工具
    :return: 无
    :rtype: 无
    '''
    jieba.initialize()

#输入：用户信息，分词模式 (True = 全模式)
#输出：词组的list

def fen_ci(message, cut_mode):
    '''
    分词道具
    :param message: 用户输入的信息。
    :type message: str
    :param cut_mode: 分词模式，True为全模式
    :type cut_mode: boolean
    :return: 分好词的list
    :rtype: list
    '''
    seg_list = jieba.cut(message, cut_mode)
    return list(seg_list)
