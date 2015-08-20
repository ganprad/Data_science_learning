
#!usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from scipy.stats import chisqprob
from numpy.linalg import lstsq, inv
from numpy import dot
from statsmodels.discrete.discrete_model import Logit


def load_dataset(dataset_loc):
# Importing data
    loansData = pd.read_csv(dataset_loc)

# Making Interest.Rate into floats
    loansData['Interest.Rate_pc_floats'] = loansData[
        'Interest.Rate'].map(lambda x: float(x.split("%")[0]))

# Choosing midpoints of categorical variables that represent the FICO.Range
    loansData['FICO.Score'] = loansData['FICO.Range'].map(
        lambda x: ((int(str(x).split("-")[0]) + int(str(x).split("-")[1])) / 2.0))

# Removing "months" string from Loan.Length
    loansData["Loan.Length_months"] = loansData[
        "Loan.Length"].map(lambda x: int(x.split(" ")[0]))

# Removing % from debt to income ratio
    loansData['Debt.To.Income.Ratio_pc'] = loansData[
        'Debt.To.Income.Ratio'].map(lambda x: float(x.split("%")[0]))
    return loansData


def preprocess(df, input_name, ind_vars):
    """Writes out int(bool) array with appropriate mapped condition

    for Y1: interest rate(y) < 12.0 will be 1.0 else 0.0 
    for Y2: interest rate(y) > 15.0 will be 1.0 else 0.0 

    """
    #df[output_name] = df[input_name].map(lambda x: int(bool(x>=12.0)))
    y = np.array(df[input_name])
    Y1 = (y < 12.0).astype(np.float)
    Y2 = (y > 15.0).astype(np.float)
    X = np.matrix(df[ind_vars])
    ones = np.ones(X.shape[0])
    X = np.column_stack([X, ones])
    return X, Y1, Y2


def SM_logit(X, y):
    """Computing logit function using statsmodels Logit and 
    output is coefficient array."""
    logit = Logit(y, X)
    result = logit.fit()
    coeff = result.params
    return coeff


def logistic_distribution(X, **kwargs):
    """Creates a logistic regression plot"""

    coeffs = np.array(kwargs.values())
    px = 1 / (1 + np.exp(-dot(X, coeffs.T)))
    return px


def logistic_func(fico, loanamt, **kwargs):
    """
    p1 = logistic_func(720.0,10000.0,sm_coeffs = SM_logit(X,y1))

    will calculate the probability for of a getting a loan with interest
    rate < 12%(encoded in y1 from the preprocessing step), for a 
    specified fico score and loan amount.

    """
    coeffs = np.array(kwargs.values())
    X = np.array([np.float(fico), np.float(loanamt), 1.0])
    p = 1 / (1 + np.exp(-dot(X, coeffs.T)))
    return p


def pred(p):
    """Prediction is True is p>=0.7 else it is False"""
    if p >= 0.7:
        return float(True)
    else:
        return float(False)


def check(X, **kwargs):
    """
    Checks specified "fico" and "loanamt" values or 

    Random FICO and Loan Amount generated between min and max values 
    of the dataset.

    """
    random_fico = np.float(
        np.random.random_integers(np.min(X[:, 1]), np.max(X[:, 1])))
    random_loanamt = np.float(
        np.random.random_integers(np.min(X[:, 0]), np.max(X[:, 0])))

    if kwargs:

        p = logistic_func(np.float(kwargs['fico']), np.float(
            kwargs['loanamt']), sm_coeffs=SM_logit(X, y1))

        if pred(p) == 1.0:
            print("Loan will be approved for FICO score {0} and Loan Amount {1}".format(
                np.int(kwargs['fico']), np.int(kwargs['loanamt'])))
        else:
            print("Loan will not be approved for FICO score {0} and Loan Amount {1}".format(
                np.int(kwargs['fico']), np.int(kwargs['loanamt'])))

    else:

        p = logistic_func(
            random_fico, random_loanamt, sm_coeffs=SM_logit(X, y1))

        if pred(p) == 1.0:
            print("Loan will be approved for FICO score {0} and Loan Amount {1}".format(
                np.int(random_fico), np.int(random_loanamt)))
        else:
            print("Loan will not be approved for FICO score {0} and Loan Amount {1}".format(
                np.int(random_fico), np.int(random_loanamt)))


if __name__ == '__main__':

    dataset_loc = 'https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv'

    # Load Dataset and perform prelimnary cleanup
    loansData = load_dataset(dataset_loc)

# Selecting a few of the variables
    Vars = ['Amount.Requested', 'Amount.Funded.By.Investors', 'Interest.Rate_pc_floats',
            'Debt.To.Income.Ratio_pc',
            'FICO.Score']

    input_name = Vars[2]
    ind_vars = [Vars[1], Vars[4]]

    # Logistic Regression using scikit-learn
    print("Logistic Regression with scikit-learn:")
    # Preprocess step
    X, y1, y2 = preprocess(loansData, input_name, ind_vars)
    # Logistic plot
    px = logistic_distribution(X, sm_coeffs=SM_logit(X, y1))
    # FICO score plot where FICO score = X[:,1]
    # Setting plot limits
    plt.axis([np.min(X[:, 1]), np.max([X[:, 1]]), 0.0, 1.0])
    plt.xlabel('FICO.Score')
    plt.ylabel('Probability')
    sns.regplot(np.ravel(X[:, 1]), np.ravel(
        px), logistic=True, color=None, scatter_kws={'color': 'r'})
    plt.savefig('FicoScoreVsProbability.png')
    # Prediction function
    check(X, fico=720.0, loanamt=10000.0)
    check(X)
