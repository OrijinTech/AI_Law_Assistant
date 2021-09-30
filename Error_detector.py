#This file is used to detect input errors.
import pycorrector
#from pycorrector.macbert.macbert_corrector import MacBertCorrector

def make_correction(message):
    '''
    寻找并纠正用户信息内的错误。
    :param message: 用户输入信息
    :type message: str（中文）
    :return: 纠正过后的用户信息
    :rtype: str（中文）
    '''
    correct_message, detail = pycorrector.correct(message)
    #print(correct_message)

    return correct_message



