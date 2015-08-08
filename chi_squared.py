
#!/usr/bin/env python

from scipy import stats
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the reduced version of the Lending Club Dataset
loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
# Drop null rows
loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])


dtype = np.dtype([("name",float),("count",float)])
#array = np.fromiter(freq.iteritems(),dtype=dtype)
#array = np.fromiter(freq.iteritems(),dtype=dtype,len(freq))

#Fast/memory efficient dict to ndarry
# = np.fromiter(freq.iteritems(),dtype=dtype)
#np.sctypeDict.keys()
#plt.plot(t[:,0],t[:,1])
#chi_vec = stats.chisqprob(chisq,len(dict(t).keys())-1)
plt.xlabel('Open.CREDIT.Lines')
plt.ylabel('Counts')
plt.grid(True)
plt.plot(freq.keys(),freq.values())
plt.savefig('distributionplot.png')
#plt.show()

exp_val = np.mean(freq.values())
df = len(freq.values())

#sum(observed value-expected value)^2/(expected value)

chisq = np.sum((1/exp_val)*(np.array(freq.values())-exp_val)**2)
p = stats.chisqprob(chisq,df)

x = np.linspace(0.9,1.1,1000)
chi_vec = stats.chisqprob(chisq,x*chisq)
plt.figure()
plt.plot(x,chi_vec)
plt.xlabel('x = df/chisq')
plt.ylabel('chisqProb')
plt.savefig('ChiSquareProbabilityPlot.png')
#plt.show()

print("Chi-Squared Value = {0}\np-value = {1} ".format(chisq , p))

