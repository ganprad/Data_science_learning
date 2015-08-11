
#!usr/bin/env

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from scipy.stats import chisqprob
from numpy.linalg import lstsq
from numpy import dot, invert as inv

#Importing data
loansData = pd.read_csv(
    'https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

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

# Selecting a few of the variables
Vars = ['Amount.Requested', 'Amount.Funded.By.Investors', 'Interest.Rate_pc_floats',
        'Debt.To.Income.Ratio_pc',
        'FICO.Score']


# PairGrid class in Seaborn
g = sns.PairGrid(loansData, vars=Vars)
g = g.map_lower(sns.kdeplot)
g = g.map_diag(plt.hist)
g = g.map_upper(plt.scatter, s=5)
g.savefig('scatterplot.pdf')

# Setting up variables for Linear Regression
# InterestRate = b + a1(FICOScore) + a2(LoanAmount)

intrate = loansData['Interest.Rate_pc_floats']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

y = np.matrix(intrate).transpose()
x1 = np.matrix(loanamt).transpose()
x2 = np.matrix(fico).transpose()


# Linear regression using scikit-learn
print("Linear Regression with scikit-learn:")

# Setting up data for analysis
X = np.column_stack([x1, x2])

X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = LinearRegression()
# Fitting model to data
regressor.fit(X_train, y_train)

Rsq = regressor.score(X_test, y_test)

# Testing model against data
print("R-squared with scikit-learn LinearRegression: {}".format(Rsq))

#Linear regression using numpy.linalg routines 
ones = np.ones(X_train.shape[0])
X_np = np.column_stack([ones,X_train])
ones = np.ones(X_test.shape[0])
X_nptest = np.column_stack([ones,X_test])

print("Linear regression using numpy routines: ")

#Parameter column vector = ((X.T * X)^-1) * (X.T * y))

Betas = dot(inv(dot(X_np.T,X_np)),dot(X_np.T,y_train))

print("Betas with dot and inverse: \n{}\n".format(Betas))

#Generating predictions from Betas vector
y_predict = np.dot(Betas.T,X_nptest.T)

squares_sum = np.sum(np.ravel(y_test-np.mean(y_test))**2)
residuals_sum = np.sum(np.power(np.ravel(y_test-y_predict.T),2))
Rsq = 1 - residuals_sum/squares_sum

print("R-squared: {}\n".format(Rsq2))
      
#Using numpy.linalg.lstsq

print("Betas with numpy.linalg: \n{}\n".format(lstsq(X_np,y_train)[0]))
      

#Goodness of fit test
chisq = np.sum(np.ravel(y_test)-np.ravel(y_predict)/np.ravel(y_predict))

p = chisqprob(chisq,(len(y_test)-1))
print("Goodness of fit:")
print("Chi-Sq:{0} , P-Value:{1}".format(chisq,p))