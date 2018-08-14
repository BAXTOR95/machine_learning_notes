# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Module import backwardelimination as be

# Importing the dataset
dataset = pd.read_csv(
    "/home/baxtor95/ML_Course/Projects/Part 2/S5/50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
"""import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()"""

# Automazing the BE process
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
# Preparing the model by just the p-value only
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_opt = be.backwardEliminationP(X_opt, y, 0.05)

# Preparing the model by the p-value and R-Squared
X_opt_R = X[:, [0, 1, 2, 3, 4, 5]]
X_opt_R = be.backwardEliminationPR(X_opt_R, y, 0.05)

# Splitting the dataset into the Training set and Test set
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_opt, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
regressor_opt = LinearRegression()
regressor_opt.fit(X_train_2, y_train_2)

# Predicting the Test set results
y_pred_2 = regressor_opt.predict(X_test_2)

# Splitting the dataset into the Training set and Test set
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
    X_opt_R, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
regressor_opt_2 = LinearRegression()
regressor_opt_2.fit(X_train_3, y_train_3)

# Predicting the Test set results
y_pred_3 = regressor_opt_2.predict(X_test_3)
