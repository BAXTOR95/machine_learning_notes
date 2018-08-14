# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(
    '/home/baxtor95/ML_Course/Projects/Part 7/Restaurant_Reviews.tsv',
    delimiter='\t', quoting=3)


def eval_performance(cm):
    tp = cm[0, 0]
    tn = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]
    accuracy = (tp + tn)/(tp + tn + fp + fn)
    precision = tp/(tp + fp)
    recall = tp/(tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    print(f"True Positives: {tp}\nTrue Negatives: {tn}\n" +
          f"False Positives: {fp}\nFalse Negatives: {fn}\n")
    print(f"Accuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\n" +
          f"F1 Score: {f1_score}")

# Cleaning the text
import re
import nltk
###############################################################################
#                                 Optional Code
# import earthy
# Downloading all ntlk popular packages
# nltk.download('popular', halt_on_error=False)
# Downloading all ntlk packages with earthy
# path_to_nltk_data = '/home/baxtor95/nltk_data/'
# earthy.download('all', path_to_nltk_data)
#
###############################################################################
nltk.download('stopwords', halt_on_error=False)
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer as PS
corpus = []

for i in range(0, 1000):
    review = re.sub('[^A-Za-z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PS()
    review = [ps.stem(word) for word in review if word
              not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer as CV
cv = CV(max_features=1500)  # Keeping only the 1500 most used words
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                    random_state=0)

# Using Naives Bayes model
# Fitting the Classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\nEvaluating performance for Naives Bayes model")
eval_performance(cm)

# Using Decision Tree Classification model
# Fitting the Classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nEvaluating performance for Decision Tree Classification model")
eval_performance(cm)

# Using Random Forest Classification model
# Fitting the Classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(
    n_estimators=300, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nEvaluating performance for Random Forest Classification model")
eval_performance(cm)
