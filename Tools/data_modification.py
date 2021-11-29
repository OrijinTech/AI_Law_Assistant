import json


def add_pattern(sentence, law_type):
    with open("D:\AI_Law_Assistant\Language_Data\Law_Data.json", "r", encoding="utf8") as file:
        # returning the Json object
        data = json.load(file)
        for intent in data["law_database"]:
            # For each pattern inside each intent, we want to tokenize the words that are in sentences.
            if intent["law_type"] == law_type:
                intent["law_keywords"].append(sentence)
    with open("D:\AI_Law_Assistant\Language_Data\Law_Data.json", "w", encoding="utf8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


