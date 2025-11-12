import pandas as pd
import numpy as np
pd.set_option("display.max_columns", None)

from sklearn.impute import KNNImputer

#################
# Categorical scaling value
categorical_scaler = 0.66
#################

# Load in data 
df = pd.read_excel("../Data/dat.xlsx")

df.head()
df.columns
df.shape

# drop a few extra cols
df.drop(columns = ['bess_hard1', 'bess_hard2', 'bess_hard3', 'loc', 'add_adhd', 'ld_dyslexia', 'migraines', 'anxiety', 'depression'], inplace=True)

# check value counts
categoricals = ['sex1f', 'number_prior_conc', 'exercise_since_injury', 'current_sleep_problems']
for col in categoricals:
    print(df[col].value_counts())

##############
## Separate the recovery outcomes from the actual features used for similarity comparison
X = df.drop(columns=['time_sx', 'PPCS', 'time_rtp'])
y = df[['time_sx', 'PPCS', 'time_rtp']]

X.shape

#############
## Check for missingness 
X.isna().sum()
X.describe()

# KNN imputer
imputer = KNNImputer(n_neighbors=40)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

# Check cols
for col in X.columns:
    print(X[col].value_counts())

# round all cols except age to nearest whole num
col_to_round = ['time_since_injury', 'sex1f', 'number_prior_conc', 'exercise_since_injury', 
    'headache_severity', 'current_sleep_problems', 'BESS_total', 'HBI_total']

for col in col_to_round:
    X[col] = np.round(X[col])

X['age'].head()
X['BESS_total'].head()

##############
## Min-Max Scaling (just for X, don't care for y)
X_mins = X.min()
X_ranges = (X.max() - X.min())

for col in X.columns:
    X[col] = (X[col] - np.min(X[col])) / (np.max(X[col]) - np.min(X[col]))

# Summarize 
X.describe()

# down-weight categoricals - contributing too much to distance metrics
X[categoricals].describe()

X[categoricals] = X[categoricals]*categorical_scaler

X[categoricals].describe()

# cut out significant outliers in outcome var
outcome_threshold = 100
y.loc[y['time_sx'] > outcome_threshold, 'time_sx'] = np.nan
y.loc[y['time_rtp'] > outcome_threshold, 'time_sx'] = np.nan

y['time_sx'].max()
