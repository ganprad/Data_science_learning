
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import numpy as np
import seaborn as sns

#Loading Lending Club loan data

loansData = pd.read_csv("https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv")

#Cleaning data
loansData.dropna(inplace=True)

#Getting desired plotting data
Data = loansData[['Amount.Requested','Amount.Funded.By.Investors']]

#Function for calculating z values
def Z_value(Data):
    Z_values = np.zeros(Data.shape)
    for i,item in enumerate(Data.columns):
        x = np.array(Data[Data.columns[i]])
        Z_values[:,i] = (x-np.mean(x))/np.std(x)
    return Z_values


#Using Seaborn to plot a combined boxplot
plt.figure()
g1 = sns.boxplot(data=Data,palette='Set2')
plt.savefig('boxplot.png',bbox_inches="tight")
#plt.show()
plt.close()

#Joint plot
g = sns.jointplot(x='Amount.Requested',y='Amount.Funded.By.Investors',
              data=Data,
              marginal_kws=dict(bins=50, rug=True),
              size=10,
              )
rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
g.annotate(rsquare,template="{stat}: {val:.4f}",
          stat="$R^2$",loc="upper right",
          fontsize = 12)
#plt.show()
g.savefig('joinplot.png')

z = Z_value(Data)

#QQplot using stats.probplot
f, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True,sharey=True)
sns.despine(left=True)
plt.subplot(2,1,1)
graph1 = stats.probplot(z[:,0], dist="norm", plot=plt)
plt.title('Amount.Requested Probability Plot')
plt.ylabel('Ordered Values')
plt.xlabel('')
plt.subplot(2,1,2)
graph2 = stats.probplot(z[:,1], dist="norm", plot=plt)
plt.title('Amount.Funded.By.Investors Probability Plot')
plt.ylabel('Ordered Values')
#plt.show()
plt.savefig('probability_plots.png')
plt.close()