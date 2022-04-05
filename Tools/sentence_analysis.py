

import jieba
from jieba import posseg
from jieba import analyse
import support_fnc


def sentence_structure(sentence, word_elem=''):
    """
    用于分析词性, https://github.com/duoergun0729/nlp/blob/master/%E4%BD%BF%E7%94%A8Jieba%E8%BF%9B%E8%A1%8C%E4%B8%AD%E6%96%87%E8%AF%8D%E6%80%A7%E6%A0%87%E6%B3%A8.md
    :param word_elem:词性,具体请看github链接(str)
    :param sentence:句子(str)
    :return:list
    """
    split_sent = posseg.cut(sentence.strip())
    out_sent = []
    out_picked = []
    for w, tag in split_sent:
        word_tuple = (w, tag)
        out_sent.append(word_tuple)
    for r in out_sent:
        if r[1] == word_elem:
            out_picked.append(r[0])
        elif word_elem == '':
            out_picked = out_sent
    return out_picked


def key_extraction(sentence, rank_num=5):
    rank_list = jieba.analyse.extract_tags(sentence, rank_num)
    return rank_list


def calc_answer(json_file, user_sent, category):
    out_ans = ''
    data = support_fnc.open_file(json_file)
    for response in data["responses"]:
        ans_list = sentence_structure(response, 'ns')
        user_list = sentence_structure(user_sent, 'ns')
        #for key in ans_list:
    return out_ans


print(sentence_structure("我的儿子去年在上海买了一辆车，但是被偷了。", 'ns'))
# print(key_extraction("我的儿子去年买了一辆车，但是被偷了。"))
