"""This module contains two functions for applying
a backward elimination to a model dataset
"""
import statsmodels.formula.api as sm
import numpy as np


def backwardEliminationP(x, y, sl):
    """Function that applies a Backward Elimination
    on a model with p-values only

    Arguments:
        x {Array} -- The Predictors values
        y {Array} -- The Dependent Variable
        sl {Float} -- The Significance Level

    Returns:
        Array -- The New Optimized Model
    """
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x


def backwardEliminationPR(x, y, SL):
    """Function that applies a Backward Elimination
    on a model with p-values and Adjusted R Squared

    Arguments:
        x {Array} -- The Predictors values
        y {Array} -- The Dependent Variable
        SL {Float} -- The Significance Level

    Returns:
        Array -- The New Optimized Model
    """
    numVars = len(x[0])
    temp = np.zeros((50, 6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if regressor_OLS.pvalues[j].astype(float) == maxVar:
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    temp_regressor = sm.OLS(y, x).fit()
                    adjR_after = temp_regressor.rsquared_adj.astype(float)
                    if adjR_before >= adjR_after:
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
