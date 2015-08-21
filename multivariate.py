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
import statsmodels.api as sm
import statsmodels.formula.api as smf


from IPython.core.display import HTML
import zipfile


def load_dataset(file_loc,**kwargs):
    z = zipfile.ZipFile(file_loc)
    df = pd.read_csv(z.open('LoanStats3d.csv'),skiprows=[0],nrows=kwargs['nrows'])
    return df

def preprocess(df):
    
    df['logannual_inc']=df['annual_inc'].map(lambda x: np.log1p(np.float(str(x))))
    df['annual_inc'] = df['annual_inc'].map(lambda x: np.float(str(x)))
    df['int_rate']=df['int_rate'].map(lambda x: np.float(str(x).strip().split('%')[0]))
    df['logint_rate']=df['int_rate'].map(lambda x: np.log1p(np.float(str(x).strip().split('%')[0])))
    return df

def estimator(s,**kwargs):
    est = smf.ols(formula=s,data=kwargs['data']).fit()
    return est

def short_summary(est):
    return HTML(est.summary().tables[1].as_html())

if __name__ == '__main__':
    file_loc = 'Data/LoanStats3d.csv.zip'
    
    #Creating a Dataframe from file

    df = load_dataset(file_loc,nrows=1500)

    # Preprocess step
    
    df = preprocess(df)
    
    formula='int_rate ~ logannual_inc + C(home_ownership) + logannual_inc*C(home_ownership)'
    est = estimator(formula,data=df)
    param = est.params
    income_linspace = np.linspace(df.logannual_inc.min(), df.logannual_inc.max(), 100)

    plt.scatter(df.logannual_inc,df.int_rate,alpha=0.3)
    plt.xlabel('LogIncome')
    plt.ylabel('InterestRate')
    plt.title(formula)
    plt.axis([np.min(df.logannual_inc), np.max(df.logannual_inc), np.min(df.int_rate), np.max(df.int_rate)])
    #OWN
    plt.plot(income_linspace,param[0] + param[1]*1 + param[2]*0+
             param[3]*income_linspace+ param[4]*income_linspace,'r')
    #RENT
    plt.plot(income_linspace,param[0]+param[1]*0 + param[2]*1+
             param[3]*income_linspace+param[5]*income_linspace,'g')

    plt.savefig('IntRateVsLogIncome.pdf',bbox_inches='tight')