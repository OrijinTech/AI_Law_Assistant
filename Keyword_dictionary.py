# 这个文件是用来整理，编辑关键词词典的
import jieba

def dictionary_setup(filename):
    jieba.load_userdict(filename)

def add_to_dictionary(new_word):
    jieba.add_word(new_word)

def delete_from_dictionary(word_for_deletion):
    jieba.del_word(word_for_deletion)
