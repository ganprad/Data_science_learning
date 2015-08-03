
# !/usr/bin/env python

import pandas as pd
from scipy import stats

data = '''Region, Alcohol, Tobacco
North, 6.47, 4.03
Yorkshire, 6.13, 3.76
Northeast, 6.19, 3.77
East Midlands, 4.89, 3.34
West Midlands, 5.63, 3.47
East Anglia, 4.52, 2.92
Southeast, 5.89, 3.20
Southwest, 4.79, 2.71
Wales, 5.27, 3.53
Scotland, 6.08, 4.51
Northern Ireland, 4.02, 4.56'''


data = data.split('\n')
data = [i.split(', ') for i in data]
column_names = data[0]
data_rows = data[1:]
df = pd.DataFrame(data_rows,columns=column_names)

df['Alcohol'] = df['Alcohol'].astype('float')
df['Tobacco'] = df['Tobacco'].astype('float')

# (mean,median,mode)
def range_(x):
    return max(x) - min(x)


Al_summary = (df['Alcohol'].mean(),
                   df['Alcohol'].median(),
                   stats.mode(df['Alcohol'])[0][0],
                   range_(df['Alcohol']),
                   df['Alcohol'].var(),
                   df['Alcohol'].std()
                  )
To_summary = (df['Tobacco'].mean(),
              df['Tobacco'].median(),
              stats.mode(df['Tobacco'])[0][0],
              range_(df['Tobacco']),
              df['Tobacco'].var(), 
              df['Tobacco'].std()
              )
header = ['mean','median','mode','range','variance',
          'standard deviation']
print("Alcohol Data Summary:")
summary_list = [header[i]+" is {:.4f}".format(item) for i,item in enumerate(Al_summary)]
for i in summary_list:
    print i

    
print("\nTobacco Data Summary:")
summary_list = [header[i]+" is {:.4f}".format(item) for i,item in enumerate(To_summary)]

for i in summary_list:
    print i