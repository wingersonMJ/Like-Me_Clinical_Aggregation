import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer

# Load in data 
df = pd.read_excel("../Data/dat.xlsx")

df.head()
df.columns
df.shape

# drop a few extra cols
df.drop(columns = ['bess_hard1', 'bess_hard2', 'bess_hard3'], inplace=True)

# check value counts
categoricals = ['sex1f', 'loc', 'number_prior_conc', 'add_adhd', 'ld_dyslexia', 'anxiety', 'depression', 'migraines', 'exercise_since_injury', 'headache_severity', 'current_sleep_problems']
for col in categoricals:
    print(df[col].value_counts())

# combine cols with low value counts
df['learn_disord'] = np.where((df['ld_dyslexia'] == 1) | (df['add_adhd'] == 1), 1, 0)
df['anx_dep'] = np.where((df['anxiety'] == 1) | (df['depression'] == 1), 1, 0)

# drop low value count cols
df.drop(columns = ['ld_dyslexia', 'add_adhd', 'anxiety', 'depression'], inplace=True)

##############
## Separate the recovery outcomes from the actual features used for similarity comparison
X = df.drop(columns=['time_sx', 'PPCS', 'time_rtp'])
y = df[['time_sx', 'PPCS', 'time_rtp']]

#############
## Check for missingness 
X.isna().sum()
X.iloc[:,0:5].describe()
X.iloc[:,5:8].describe()
X.iloc[:,8:].describe()

# KNN imputer
imputer = KNNImputer(n_neighbors=20)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Check cols
for col in X.columns[:5]:
    print(X[col].value_counts())

for col in X.columns[5:10]:
    print(X[col].value_counts())

# round all cols except age to nearest whole num
col_to_round = ['time_since_injury', 'sex1f', 'loc', 'number_prior_conc',
       'migraines', 'exercise_since_injury', 'headache_severity',
       'current_sleep_problems', 'BESS_total', 'HBI_total', 'learn_disord',
       'anx_dep']

for col in col_to_round:
    X[col] = np.round(X[col])

X['age'].head()
X['BESS_total'].head()

##############
## Min-Max Scaling (just for X, don't care for y)
for col in X.columns:
    X[col] = (X[col] - np.min(X[col])) / (np.max(X[col]) - np.min(X[col]))

# Summarize 
X.iloc[:,0:5].describe()
X.iloc[:,5:8].describe()
X.iloc[:,8:].describe()