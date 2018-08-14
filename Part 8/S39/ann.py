# Artificial Neural Network


def eval_performance(cm):
    """eval_performance (cm)

    Function that calculates the performance of a ML Model
    by getting the Accuracy, Precision, Recall and F1 Score values

    Arguments:
        cm {List} -- The Confusion Matrix
    """
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

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website:
# https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# Nodes in the hidden layer = (Number of nodes in the input layer + 1)/2
# relu correspond to the rectifier function
classifier.add(Dense(activation="relu", input_dim=11,
                     units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1,
                     kernel_initializer="uniform"))

# Compiling the ANN
# adam correspond to a stochastic gradient descent algorithm
"""logaritmic loss function
   If dependent variable has a binary outcome then binary_crossentropy
   If dependent variable has more than 2 outcomes like 3 categories then
   the logaritmic loss function is called categorical_crossentropy
"""
classifier.compile(
    optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("\nEvaluating performance for ANN model")
eval_performance(cm)
