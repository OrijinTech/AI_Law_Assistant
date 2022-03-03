import docx
import support_fnc
import unicodedata

list_of_words_del = ["\u3000"]


def extract_paragraph(filename):
    doc = docx.Document(filename)
    par_list = []
    for para in doc.paragraphs:
        par_list.append(para.text)
    return par_list


def format_para(par_list, list_of_del_word):
    unicode_list = [unicodedata.normalize('NFKC', line) for line in par_list]
    updated_list = []
    for par in unicode_list:
        updated_list.append(par.strip())
    print(updated_list)
    for par2 in updated_list:
        if par2 == '':
            updated_list.remove(par2)
        for w in list_of_words_del:
            if w in par2:
                par2.replace(w, '')
    return updated_list


def add_to_(par_list):
    return None


def delete_from_par(word, paragraph):
    set_par = paragraph
    set_par.replace(word, "")
    return set_par


def delete_from_list(word, par_list):
    ret_list = par_list
    for par in ret_list:
        if word in par:
            ret_list.remove(par)
        # elif len(par) < 10:
        #     ret_list.remove(par)
    return ret_list


def updated_list_of_para():
    l1 = extract_paragraph("../Language_Data/docx_data/中华人民共和国义务教育法(2018修正)(FBM-CLI-1-328270).docx")
    print(l1)
    l1 = format_para(l1, list_of_words_del)
    print(l1)


# updated_list_of_para()

txy = '  第一章 总  则\n  第二章 学  生\n  第三章 学  校\n  第四章 教  师\n  第五章 教育教学\n '
print(txy)
txy.replace("\n", '')
print(txy)