""" What were the most frequent themes that emerged from survey responses
to what progress for women’s human rights looks like in 10 years?

How do these themes differ by segments of the survey population (age, gender, etc.)?

Data:
- globalcount_data.csv contains all the raw data (- 7500 filtered non-responses)
- Global Count Data Dictionary.xlsx contains info on the data structure
- Question ID (QID) # 9 is the free response answer.
- Long, med, and short versions of the survey all had this question.

-> column code "progress_10_years"

Goal:
(1) Generate wordcloud for all survey responses
(2) Generate wordcloud for other segments of the survey population and
see look into how they differ.

Tutorial Reference: https://towardsdatascience.com/generate-meaningful-word-clouds-in-python-5b85f5668eeb

"""

import os
import sys
import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

#additional stop words
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP_WORDS

BASE_PATH = os.getcwd()
GLOBAL_COUNT_CSV_FILE = "globalcount_data.csv"
MORE_STOPWORDS = ["nan", "will", "will", "without", "today", "now", "much", "come", "years", "dont", "know", "want", "able", "thing", "women", "look"]
# remove special characters: https://thispointer.com/remove-multiple-character-from-a-string-in-python/
INVALID_CHARACTER_LIST = ["?", ".", ",", "\d+", "%", "(", ")"]
INVALID_CHARACTERS = '[' +  ''.join(INVALID_CHARACTER_LIST) +  ']'


def save_cleaned_data(df, resave=False, info=None):
    file_path = f"{BASE_PATH}/cleaned_data/{info}.csv"
    file_exists = os.path.exists(file_path)
    if not file_exists or resave:
        df.to_csv(file_path)
        print("Resaved data")

    print("data was not saved - add resave parameter to overwrite")
    return pd.read_csv (r'{}'.format(file_path))

def get_sanitized_text(df):
    # remove most invalid characters and join the responses into a long string
    text = " ".join(re.sub(INVALID_CHARACTERS, "", review) for review in df.str.lower())
    text = text.replace("\\r\\n","")
    text = text.replace("'`","")

    return text

def plot_word_cloud(word_cloud):
    plt.axis("off")
    plt.figure(figsize=(40,20))
    plt.tight_layout(pad=0)
    plt.imshow(sanitized_word_cloud, interpolation='bilinear')
    plt.show()
    return


def generate_all_response_wordcloud(df):
    """Generates the wordcloud for all. """
    df_progress_response = df["progress_10_years_tr"]
    non_response_count = df_progress_response.isna().sum()
    total_count = len(df_progress_response.index)
    # print(f"Total records: {total_count}")
    non_response_percent = (non_response_count / total_count) * 100
    # print("Percent of non-responses: {}".format(non_response_percent))
    responses_with_input = df_progress_response.dropna()
    sanitized_text = get_sanitized_text(responses_with_input)
    stopwords = set(STOPWORDS)
    stopwords.update(SPACY_STOP_WORDS)
    stopwords.update(MORE_STOPWORDS)
    word_cloud =WordCloud(stopwords=stopwords, min_word_length=4, collocation_threshold=3, collocations=True, background_color="white", width=800, height=400).generate(sanitized_text)

    return word_cloud.words_



if __name__ == "__main__":
    df = pd.read_csv (r'{}/{}'.format(BASE_PATH, GLOBAL_COUNT_CSV_FILE))
    df_progress_response = save_cleaned_data(df["progress_10_years_tr"], resave=False, info="progress_responses_tr")
    generate_all_response_wordcloud(df_progress_response)
