import openpyxl
from pathlib import Path
import sentence_analysis


def read_from_xl(num_lines):
    xlsx_file = Path("/Users/mushr/PycharmProjects/AI_Law_Assistant/Language_Data/Training_raw/问题筛选和分类1.xlsx")
    wb_file = openpyxl.load_workbook(xlsx_file)
    # print(wb_file.sheetnames)  # print all sheet names
    data_sheet = wb_file['sheet0']
    for row in data_sheet.iter_rows(max_row=num_lines):
        for cell in row:
            if cell.value:
                print(cell.value)
            else:
                continue




read_from_xl(2)