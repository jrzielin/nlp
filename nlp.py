import re
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def load_dataset():
    return pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

def clean_text(dataset, range_size=1000):
    corpus = []
    for i in range(0, range_size):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    return corpus

def create_bag_of_words_model(dataset, corpus, max_features=1500):
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    return (X, y)

def split_dataset(X, y, test_size=0.20, random_state=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return (X_train, X_test, y_train, y_test)

def train_model(X_train, y_train):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    return classifier

def predict_test_results(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
    return y_pred

def make_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    score = accuracy_score(y_test, y_pred)
    print(score)

def main():
    dataset = load_dataset()
    corpus = clean_text(dataset)
    model = create_bag_of_words_model(dataset, corpus)
    split_out_dataset = split_dataset(model[0], model[1])
    classifier = train_model(split_out_dataset[0], split_out_dataset[2])
    y_pred = predict_test_results(classifier, split_out_dataset[1], split_out_dataset[3])
    make_confusion_matrix(split_out_dataset[3], y_pred)

if __name__ == '__main__':
    main()