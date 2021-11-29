import json


def add_to_file(sentence, law_type):
    with open("D:\AI_Law_Assistant\Language_Data\Law_Data.json", "r", encoding="utf8") as file:
        # returning the Json object
        data = json.load(file)
        print(data)
    with open("D:\AI_Law_Assistant\Language_Data\Law_Data.json", "w", encoding="utf8") as file:
        json.dump(data, file)
