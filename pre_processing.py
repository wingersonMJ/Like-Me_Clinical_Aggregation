import pandas as pd
import numpy as np

# Load in data 
df = pd.read_excel("../Data/dat.xlsx")

df.head()
df.columns

# drop a few extra cols
df.drop(columns = ['bess_hard1', 'bess_hard2', 'bess_hard3'], axis=1, inplace=True)

# check value counts
categoricals = ['sex1f', 'loc', 'number_prior_conc', 'add_adhd', 'ld_dyslexia', 'anxiety', 'depression', 'migraines', 'exercise_since_injury', 'headache_severity', 'current_sleep_problems']
for col in categoricals:
    print(df[col].value_counts())

# combine cols with low value counts
df['learn_disord'] = np.where((df['ld_dyslexia'] == 1) | (df['add_adhd'] == 1), 1, 0)
df['anx_dep'] = np.where((df['anxiety'] == 1) | (df['depression'] == 1), 1, 0)

# drop low value count cols
df.drop(columns = ['ld_dyslexia', 'add_adhd', 'anxiety', 'depression'], axis=1, inplace=True)

