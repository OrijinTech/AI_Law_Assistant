import sys
from AI_Assistant import Aiyu

#Creating Lists
training_set = []
output_data = []
words = []
labels = []
docs_x = []
docs_y = []
model = None

# AI Configuration for Law Popularization Project
#intent_file = "Language_Data/Law_Data.json"
intent_file = "Language_Data/temp.json"
intents = "law_database"
tags = "law_type"
patterns = "law_keywords"
language = "ch"

#AI Configuration for Personal AI Project
# intent_file = "../Language_Data/intents.json"
# intents = "intents"
# tags = "tag"
# patterns = "patterns"
# language = "en"

def main(start_var):
    aiyu = Aiyu.Aiyu(training_set, output_data, words, labels, docs_x, docs_y, model, intent_file, intents, tags, patterns, language)
    aiyu.data_processor()
    aiyu.train_model(8, 8, 120)
    aiyu.chat()
    return 0


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])
