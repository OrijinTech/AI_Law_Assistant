def cont_num(keyword) -> bool:
    '''
    检测str内是否有integer
    :param keyword: 输入词语
    :type keyword: str
    :return: 有数字或没数字
    :rtype: bool
    '''
    return any(char.isdigit() for char in keyword)

def clear_chat():
    for i in range(50):
        print("\n")
