from Tools.test_file import *
from AI_Assistant import Aiyu

#Creating Lists
training_set = []
output_data = []
words = []
labels = []
docs_x = []
docs_y = []
model = None

#AI Configuration
intent_file = "Language_Data/Law_Data.json"
intents = "law_database"
tags = "law_type"
patterns = "law_keywords"

def main(start_var):
    aiyu = Aiyu.Aiyu(training_set, output_data, words, labels, docs_x, docs_y, model, intent_file, intents, tags, patterns)
    aiyu.data_processor()
    aiyu.train_model(12, 12, 120)
    aiyu.chat()
    return 0


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    main(sys.argv[0])
