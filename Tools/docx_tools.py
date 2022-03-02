import docx


def extract_paragraph(filename):
    doc = docx.Document(filename)
    par_list = []
    for para in doc.paragraphs:
        par_list.append(para.text)
    print(par_list)
    return par_list
# extract_paragraph("../Language_Data/docx_data/中华人民共和国义务教育法(2018修正)(FBM-CLI-1-328270).docx")

def delete_paragraph(par_list):


def delete_word(paragraph):

