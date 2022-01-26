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
(1) Generate wordcloud for all survey responses (done in word_cloud.py)
(2) Generate wordcloud for other segments of the survey population and
see look into how they differ.

Notes:
- gather all data then remove rows if they don't meet a column criteria.
- combile one big dataframe and do some sort of cross comparison

Subgroups: country, gender, race, age, language

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

import seaborn as sns
sns.set_style('darkgrid')


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



def save_cleaned_subgroup_data(df, resave=False, info=None):
    """country, gender, race, age, language"""
    # master column where header is populated if that specfic row is 1 - need to allow multiple genders and multiple races
    file_path = f"{BASE_PATH}/cleaned_data/{info}.csv"
    file_exists = os.path.exists(file_path)
    if not file_exists or resave:
        # make gender column
        genders = ["gender_not_listed", "cisgender_man", "cisgender_woman", "non_conforming_or_non_binary", "gender_choose_not_to_identify", "man", "transgender_man", "transgender_woman", "woman"]
        df["gender"] = ""
        row_gender = ""
        for i in df.index:
            for g in genders:
                value = df.iloc[[i]][g].values[0]
                if value == 1:
                    row_gender = row_gender + f"+{g}"
            df.loc[[i],'gender'] = row_gender.strip("+")
            row_gender = ""
        # make race column
        race_1 = ["racial_identity_not_listed","asian","biracial_or_mixed","black_or_of_african_descent","hispanic_or_latinx","choose_not_to_identify", "indigenous","indigenous_central_or_south_american"]
        race_2 = ["middle_eastern","native_american","native_hawaiian","north_african","pacific_islander","south_asian","southeast_asian","white"]
        races = race_1 + race_2
        df["race"] = ""
        row_race = ""
        for i in df.index:
            for r in races:
                value = df.iloc[[i]][r].values[0]
                if value == 1:
                    row_race = row_race + f"+{r}"
            df.loc[[i],'race'] = row_race.strip("+")
            row_race = ""


        filtered_df = df[['iso3166', 'gender', 'race', 'age', 'language', 'progress_10_years_tr']]

        filtered_df.to_csv(file_path, index=False)
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
    plt.imshow(word_cloud, interpolation='bilinear')
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
    # dict with words and there probabilities
    word_df = pd.DataFrame(word_cloud.words_.items(), columns=['words_all', 'frequency_all'])
    plot_word_cloud(word_cloud)
    return word_df, worcloud_all


def generate_subgoup_wordcloud(subgroup):
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
    # dict with words and there probabilities
    word_df = pd.DataFrame(word_cloud.words_.items(), columns=['words_all', 'frequency_all'])
    return word_df, worcloud_all

    # def generate_all_subgroup_frequency(word_frequency_df):
    #     subgroups =




if __name__ == "__main__":
    df = pd.read_csv (r'{}/{}'.format(BASE_PATH, GLOBAL_COUNT_CSV_FILE))
    df_progress_response = save_cleaned_data(df["progress_10_years_tr"], resave=False, info="progress_responses_tr")
    word_frequency_df, wordcloud_all = generate_all_response_wordcloud(df_progress_response)
    df_progress_subgroup_response = save_cleaned_subgroup_data(df, resave=False, info="subgroup_responses")
    print(df_progress_subgroup_response.info())
    print(df_progress_subgroup_response.describe(include='object'))

    for column in df_progress_subgroup_response.select_dtypes(include='object'):
        if df_progress_subgroup_response[column].nunique() < 40:
            sns.countplot(y=column, data=df_progress_subgroup_response)
            plt.show()


    # generate_all_subgroup_frequency(word_frequency_df)
