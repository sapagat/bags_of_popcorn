import pandas as pd
import csv
from bs4 import BeautifulSoup
import re
import nltk
try:
    nltk.data.find('corpora/stopwords.zip')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

MOST_FREQUENT = 5000
HEADER_ROW = 0
DELIMITER = '\t'

def read_dataset(name):
    path = 'data/' + name
    return pd.read_csv(
        path,
        header=HEADER_ROW,
        delimiter=DELIMITER,
        quoting=csv.QUOTE_NONE
    )

def process_reviews(reviews):
    all_reviews = reviews['review']
    clean_reviews = []
    for i, review in enumerate(all_reviews):
        if((i+1) % 1000 == 0 ):
            print("Review %d of %d\n" % (i+1, len(all_reviews)))
        clean_reviews.append(process_review(review))
    return clean_reviews

def process_review(review):
    without_html = remove_html_tags(review)
    letters_only = remove_all_but_letters(without_html)
    words = tokenize(letters_only)
    return ' '.join(filter_stopwords(words, stopwords.words('english')))

def remove_html_tags(text):
    return BeautifulSoup(text, features='lxml').get_text()

def remove_all_but_letters(text):
    search_pattern = '[^a-zA-Z]'
    replace_with = ' '
    return re.sub(search_pattern, replace_with, text)

def tokenize(text):
    return text.lower().split()

def filter_stopwords(words, stopwords):
    result = []
    for word in words:
        if word in stopwords:
            continue

        result.append(word)
    return result

def build_bag_of_words(texts):
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer = None,
        preprocessor = None,
        stop_words = None,
        max_features = MOST_FREQUENT
    )
    return {
        'features': vectorizer.fit_transform(texts).toarray(),
        'vocabulary': vectorizer.get_feature_names()
    }
