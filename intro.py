import numpy as np
import pandas
import pandasql
import matplotlib.pyplot as plt
import datetime
import csv
import scipy
import scipy.stats
import statsmodels.api as sm
import sys
from ggplot import *
import itertools

df = pandas.read_csv("turnstile_data_master_with_weather.csv")

plt.figure()
bins = 25
alpha = 0.75
df[df['rain']==0]['ENTRIESn_hourly'].hist(bins = bins, alpha=alpha)
df[df['rain']==1]['ENTRIESn_hourly'].hist(bins = bins, alpha=alpha)
plt.suptitle('Histogram of ENTRIESn_hourly')
plt.ylabel('Frequency')
plt.xlabel('ENTRIESn_hourly')
plt.legend(['no rain', 'rain'])
plt.grid(True)
plt.show()

print scipy.stats.shapiro(df[df['rain']==0]['ENTRIESn_hourly'])
print scipy.stats.shapiro(df[df['rain']==1]['ENTRIESn_hourly'])

with_rain_mean = np.mean(df[df['rain']==1]['ENTRIESn_hourly'])
without_rain_mean = np.mean(df[df['rain']==0]['ENTRIESn_hourly'])
U,p = scipy.stats.mannwhitneyu(df[df['rain']==1]['ENTRIESn_hourly'],
                               df[df['rain']==0]['ENTRIESn_hourly'])
print 'Mean with rain :',with_rain_mean, '\nMean without rain :', without_rain_mean, '\nU :', U, '\n2*p:', 2*p

def compute_r_squared(data, predictions):
    n = ((data - predictions)**2).sum()
    d = ((data - data.mean())**2).sum()
    r_squared = 1 - n/d
    return r_squared

def linear_regression(features, values):
    features = sm.add_constant(features)
    model = sm.OLS(values,features)
    results = model.fit()
    intercept = results.params[0]
    params = results.params[1:]
    return intercept, params

def predictions(dataframe, features):
    # Values
    values = dataframe['ENTRIESn_hourly']
    # Perform linear regression
    intercept, params = linear_regression(features, values)
    predictions = intercept + np.dot(features, params)
    return predictions, params

# List with almost all features. Following Ex 5 on Set 3, EXITSn_hourly is not being used.
aall_features = ['Hour','precipi',
                 'rain','fog',
                 'meanwindspdi',
                 'meantempi',
                 'meanpressurei',
                 'meandewpti' ]

# multiple variables to log results
i = 0
t_feat = [] # features for test i - includes dummy
t_para = [] # parameters for test i
t_pred = [] # predictions for test i
t_rsqu = [] # r2 for test i
t_subs = [] # features subset for test i
# global max logs and counter
r_max = -1
s_max = None
i_max = 0

# This brute force loop will select all combinations of some specific sizes.
# At first, not using dummies variables.
for L in range(3,(len(aall_features)+1)):
    # for each combination, runs a linear regression, loging data, preserving max
    for subset in itertools.combinations(aall_features, L):
        t_feat.append(i)
        t_para.append(i)
        t_pred.append(i)
        t_rsqu.append(i)
        t_subs.append(i)

        t_subs[i] = subset

        # adding selected features and dummy variable
        t_feat[i] = df[[subset[0]]]
        for k in range(1,len(subset)):
            t_feat[i] = t_feat[i].join(df[[subset[k]]])
        t_feat[i] = t_feat[i].join(pandas.get_dummies(df['UNIT'], prefix='unit'))

        # Perform linear regression
        t_pred[i], t_para[i] = predictions(df, t_feat[i])
        t_rsqu[i] = compute_r_squared(df['ENTRIESn_hourly'], t_pred[i])

        # Saving max for each combination size
        if r_max < t_rsqu[i]:
            r_max = t_rsqu[i]
            s_max = t_subs[i]
            i_max = i
            print '\nNew max:', i_max, 'R2:', r_max, s_max
        else:
            print i,
        i = i+1

print 'UNIT as dummy variable and :', s_max
print r_max
t_para[i_max].head(len(s_max))
r_max
