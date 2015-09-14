#!usr/bin/env python


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import zipfile


def load_dataset(file_loc, **kwargs):
    z = zipfile.ZipFile(file_loc)
    filename=file_loc.split('/')
    filename.reverse()
    df = pd.read_csv(
        z.open(filename[0].split('.zip')[0]), skiprows=[0], nrows=kwargs['nrows'],low_memory=kwargs['low_memory'])
    return df


def preprocess(df):
    df['issue_d_format'] = pd.to_datetime(df['issue_d']) 
    dfts = df.set_index('issue_d_format',drop=True) 
    return dfts




if __name__=='__main__':
    
    file_loc1 = 'Data/LoanStats3d.csv.zip'
    file_loc2 = 'Data/LoanStats3c.csv.zip'
    file_loc3 = 'Data/LoanStats3b.csv.zip'
    file_loc4 = 'Data/LoanStats3a.csv.zip'

    # Creating a Dataframe from file

    df1 = load_dataset(file_loc1,nrows=235629,low_memory=False)
    df2 = load_dataset(file_loc2,nrows=235629,low_memory=False)
    df3 = load_dataset(file_loc3,nrows=235629,low_memory=False)
    df4 = load_dataset(file_loc4,nrows=235629,low_memory=False)

    # Preprocess step

    df1 = preprocess(df1)
    df2 = preprocess(df2)
    df3 = preprocess(df3)
    df4 = preprocess(df4)
    
    df = pd.concat([df1,df2,df3,df4],axis=0)
    
    X = dfts.groupby(lambda x: x.year*100+x.month).count()['int_rate']
    sm.graphics.tsa.plot_acf(X,qstat=True,fft=True)
    plt.show()

    