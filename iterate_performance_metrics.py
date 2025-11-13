from pre_processing import X, y
from pre_processing import X_mins, X_ranges
from pre_processing import categorical_scaler, categoricals

import random 
import numpy as np
import pandas as pd
import time
from datetime import datetime
pd.set_option("display.max_columns", None)

from scipy import stats

from tableone import TableOne

import matplotlib.pyplot as plt
import seaborn as sns

# get cohort
X.head()
X.describe()
X.columns

# get outcomes 
y.head()
y.describe()

########
# Set up function for mahalanobis distance
def mahalanobis(x=None, data=None, cov=None):
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return np.sqrt(mahal)

########
# for loop for each subject to be used as the patient 
like_me_performance = pd.DataFrame(columns=[
        'pt_mahalanobis_distance_from_cohort',
        'mean_euclid_dist_to_pt', 
        'like_me_value', 
        'lm_mean_subj_diff_sx', 
        'lm_median_subj_diff_sx', 
        'lm_mean_euclid_dist_to_pt',
        'nlm_mean_subj_diff_sx', 
        'nlm_median_subj_diff_sx',
        'nlm_mean_euclid_dist_to_pt',
        'lm_mean_subj_diff_rtp', 
        'lm_median_subj_diff_rtp',
        'nlm_mean_subj_diff_rtp', 
        'nlm_median_subj_diff_rtp',
        'lm_age_diff',
        'nlm_age_diff',
        'lm_time_since_inj_diff',
        'nlm_time_since_inj_diff',
        'lm_hbi_diff',
        'nlm_hbi_diff'
    ])

start = time.time()
iteration = 0
feature_cols = X.columns
for idx, _ in X.iterrows():

    # get patient x and y
    patient = X.iloc[idx,:]
    patient_y = y.iloc[idx]

    # get cohort x and y
    cohort = X.drop(index=X.index[idx])
    cohort_y = y.drop(index=y.index[idx])

    # get patient mah distance from cohort
    pt_mahalanobis_distance_from_cohort = mahalanobis(x=patient, data=cohort)

    # get each subject's euclidean distance from patient
    for i, row in cohort.iterrows():
        euclid_dist = np.linalg.norm(row[feature_cols].to_numpy() - patient[feature_cols].to_numpy()) 
        cohort.loc[i, "euclidean_distance"] = euclid_dist

    # get mean of euclid dist
    mean_euclid_dist_to_pt = np.nanmean(cohort['euclidean_distance'])

    ###########################
    # Get like-me cohort 
    like_me_value = (10 - pt_mahalanobis_distance_from_cohort)*10
    like_me_value = np.round(like_me_value, decimals=0).astype(int)
    like_me_by = 'euclidean_distance'
    like_me_idx = cohort[like_me_by].nsmallest(like_me_value).index
    like_me_cohort = cohort.loc[like_me_idx]
    like_me_y = y.loc[like_me_idx]

    not_like_me_cohort = cohort.drop(index=like_me_idx)
    not_like_me_y = y.drop(index=like_me_idx)

    #######################
    # Demographcis - just doing Age, Time since Injury, and HBI score
    #######################
    # un-min/max scale HBI, time since inj, age
    lm_original = {}
    nlm_original = {}
    lm_original['hbi'] = (like_me_cohort['HBI_total'] * X_ranges['HBI_total']) + X_mins['HBI_total']
    nlm_original['hbi'] = (not_like_me_cohort['HBI_total'] * X_ranges['HBI_total']) + X_mins['HBI_total']

    lm_original['time'] = (like_me_cohort['time_since_injury'] * X_ranges['time_since_injury']) + X_mins['time_since_injury']
    nlm_original['time'] = (not_like_me_cohort['time_since_injury'] * X_ranges['time_since_injury']) + X_mins['time_since_injury']

    lm_original['age'] = (like_me_cohort['age'] * X_ranges['age']) + X_mins['age']
    nlm_original['age'] = (not_like_me_cohort['age'] * X_ranges['age']) + X_mins['age']

    patient_original = {}
    patient_original['age'] = (patient['age'] * X_ranges['age']) + X_mins['age']
    patient_original['hbi'] = (patient['HBI_total'] * X_ranges['HBI_total']) + X_mins['HBI_total']
    patient_original['time'] = (patient['time_since_injury'] * X_ranges['time_since_injury']) + X_mins['time_since_injury']

    # diffs 
    lm_age_diff = np.nanmean(lm_original['age']) - patient_original['age'] 
    nlm_age_diff = np.nanmean(nlm_original['age']) - patient_original['age']

    lm_time_since_inj_diff = np.nanmean(lm_original['time']) - patient_original['time']
    nlm_time_since_inj_diff = np.nanmean(nlm_original['time']) - patient_original['time']
    
    lm_hbi_diff = np.nanmean(lm_original['hbi']) - patient_original['hbi']
    nlm_hbi_diff = np.nanmean(nlm_original['hbi']) - patient_original['hbi']

    ##########################
    # time sx res
    ########################
    # Difference between the patient y and the mean of the like-me cohort values
    lm_mean_subj_diff_sx = np.nanmean(like_me_y['time_sx']) - patient_y['time_sx']
    
    # Difference between the patient y and the median of the like-me cohort values
    lm_median_subj_diff_sx = np.nanmedian(like_me_y['time_sx']) - patient_y['time_sx']

    # euclid dist to pt
    for i, row in like_me_cohort.iterrows():
        like_me_cohort.loc[i, 'lm_euclid_dist'] = np.linalg.norm(row[feature_cols].to_numpy() - patient[feature_cols].to_numpy()) 
    lm_mean_euclid_dist_to_pt = np.nanmean(like_me_cohort['lm_euclid_dist'])

    ##########################
    # Difference between the patient y and the mean of the like-me cohort values
    nlm_mean_subj_diff_sx = np.nanmean(not_like_me_y['time_sx']) - patient_y['time_sx']
    
    # Difference between the patient y and the median of the like-me cohort values
    nlm_median_subj_diff_sx = np.nanmedian(not_like_me_y['time_sx']) - patient_y['time_sx']

    # euclid dist to pt
    for i, row in not_like_me_cohort.iterrows():
        not_like_me_cohort.loc[i, 'nlm_euclid_dist'] = np.linalg.norm(row[feature_cols].to_numpy() - patient[feature_cols].to_numpy()) 
    nlm_mean_euclid_dist_to_pt = np.nanmean(not_like_me_cohort['nlm_euclid_dist'])

    ##########################
    # time rtp!
    #######################
    # Difference between the patient y and the mean of the like-me cohort values
    lm_mean_subj_diff_rtp = np.nanmean(like_me_y['time_rtp']) - patient_y['time_rtp']
    
    # Difference between the patient y and the median of the like-me cohort values
    lm_median_subj_diff_rtp = np.nanmedian(like_me_y['time_rtp']) - patient_y['time_rtp']

    ##########################
    # Difference between the patient y and the mean of the like-me cohort values
    nlm_mean_subj_diff_rtp = np.nanmean(not_like_me_y['time_rtp']) - patient_y['time_rtp']
    
    # Difference between the patient y and the median of the like-me cohort values
    nlm_median_subj_diff_rtp = np.nanmedian(not_like_me_y['time_rtp']) - patient_y['time_rtp']

    # get the important metrics
    iteration_performance = {
        'pt_mahalanobis_distance_from_cohort': pt_mahalanobis_distance_from_cohort, 
        'mean_euclid_dist_to_pt': mean_euclid_dist_to_pt, 
        'like_me_value': like_me_value, 
        'lm_mean_subj_diff_sx': lm_mean_subj_diff_sx, 
        'lm_median_subj_diff_sx': lm_median_subj_diff_sx, 
        'lm_mean_euclid_dist_to_pt': lm_mean_euclid_dist_to_pt,
        'nlm_mean_subj_diff_sx': nlm_mean_subj_diff_sx, 
        'nlm_median_subj_diff_sx': nlm_median_subj_diff_sx,
        'nlm_mean_euclid_dist_to_pt': nlm_mean_euclid_dist_to_pt,
        'lm_mean_subj_diff_rtp': lm_mean_subj_diff_rtp, 
        'lm_median_subj_diff_rtp': lm_median_subj_diff_rtp,
        'nlm_mean_subj_diff_rtp': nlm_mean_subj_diff_rtp, 
        'nlm_median_subj_diff_rtp': nlm_median_subj_diff_rtp,
        'lm_age_diff': lm_age_diff,
        'nlm_age_diff': nlm_age_diff,
        'lm_time_since_inj_diff': lm_time_since_inj_diff,
        'nlm_time_since_inj_diff': nlm_time_since_inj_diff,
        'lm_hbi_diff': lm_hbi_diff,
        'nlm_hbi_diff': nlm_hbi_diff
    }

    # save
    like_me_performance.loc[len(like_me_performance)] = iteration_performance
    like_me_performance.to_csv("../Data/like_me_performance.csv")
    print(f"iteration finished: {iteration}")
    iteration += 1
end = time.time()
print(f"{(end-start)/60} minutes to run")

################
# load back in the nresults
final_performance = pd.read_csv("../Data/like_me_performance.csv") # replace with actual file name: "../Data/like_me_performance{date}.csv"

final_performance.head()

#################
# add squared diff
squared_cols = [
    'lm_mean_subj_diff_sx',  
    'nlm_mean_subj_diff_sx', 
    'lm_median_subj_diff_sx', 
    'nlm_median_subj_diff_sx', 
    'lm_mean_subj_diff_rtp',  
    'nlm_mean_subj_diff_rtp', 
    'lm_median_subj_diff_rtp', 
    'nlm_median_subj_diff_rtp',
    'lm_age_diff',
    'nlm_age_diff',
    'lm_time_since_inj_diff',
    'nlm_time_since_inj_diff',
    'lm_hbi_diff',
    'nlm_hbi_diff'
    ]
for col in squared_cols:
    final_performance[f"{col}_sq"] = (final_performance[col])**2
    final_performance[f"{col}_sqsqrt"] = np.sqrt(final_performance[f"{col}_sq"])

###############
# summary stats
columns=[
    'like_me_value', 
    'pt_mahalanobis_distance_from_cohort', 

    'mean_euclid_dist_to_pt', 
    'lm_mean_euclid_dist_to_pt', 
    'nlm_mean_euclid_dist_to_pt', 

    'lm_mean_subj_diff_sx', 
    'nlm_mean_subj_diff_sx', 
    'lm_mean_subj_diff_sx_sq',  
    'nlm_mean_subj_diff_sx_sq', 
    'lm_mean_subj_diff_sx_sqsqrt',  
    'nlm_mean_subj_diff_sx_sqsqrt', 

    'lm_median_subj_diff_sx',
    'nlm_median_subj_diff_sx', 
    'lm_median_subj_diff_sx_sq', 
    'nlm_median_subj_diff_sx_sq',
    'lm_median_subj_diff_sx_sqsqrt', 
    'nlm_median_subj_diff_sx_sqsqrt', 

    'lm_mean_subj_diff_rtp', 
    'nlm_mean_subj_diff_rtp', 
    'lm_mean_subj_diff_rtp_sq',  
    'nlm_mean_subj_diff_rtp_sq', 
    'lm_mean_subj_diff_rtp_sqsqrt',  
    'nlm_mean_subj_diff_rtp_sqsqrt', 

    'lm_median_subj_diff_rtp', 
    'nlm_median_subj_diff_rtp', 
    'lm_median_subj_diff_rtp_sq', 
    'nlm_median_subj_diff_rtp_sq', 
    'lm_median_subj_diff_rtp_sqsqrt', 
    'nlm_median_subj_diff_rtp_sqsqrt',

    'lm_age_diff',
    'lm_age_diff_sq',
    'lm_age_diff_sqsqrt',

    'nlm_age_diff',
    'nlm_age_diff_sq',
    'nlm_age_diff_sqsqrt',

    'lm_time_since_inj_diff',
    'lm_time_since_inj_diff_sq',
    'lm_time_since_inj_diff_sqsqrt',

    'nlm_time_since_inj_diff',
    'nlm_time_since_inj_diff_sq',
    'nlm_time_since_inj_diff_sqsqrt',

    'lm_hbi_diff',
    'lm_hbi_diff_sq',
    'lm_hbi_diff_sqsqrt',

    'nlm_hbi_diff',
    'nlm_hbi_diff_sq',
    'nlm_hbi_diff_sqsqrt'
]

mytable = TableOne(final_performance, columns=columns, continuous=columns, pval=False)
print(mytable.tabulate(tablefmt = "github"))

# means and 95% CI's
reduced_columns=[
    'like_me_value', 
    'pt_mahalanobis_distance_from_cohort', 

    'mean_euclid_dist_to_pt', 
    'lm_mean_euclid_dist_to_pt', 
    'nlm_mean_euclid_dist_to_pt', 

    'lm_mean_subj_diff_sx', 
    'nlm_mean_subj_diff_sx', 
    'lm_mean_subj_diff_sx_sqsqrt',  
    'nlm_mean_subj_diff_sx_sqsqrt', 

    'lm_mean_subj_diff_rtp', 
    'nlm_mean_subj_diff_rtp', 
    'lm_mean_subj_diff_rtp_sqsqrt',  
    'nlm_mean_subj_diff_rtp_sqsqrt', 

    'lm_age_diff',
    'lm_age_diff_sqsqrt',

    'nlm_age_diff',
    'nlm_age_diff_sqsqrt',

    'lm_time_since_inj_diff',
    'lm_time_since_inj_diff_sqsqrt',

    'nlm_time_since_inj_diff',
    'nlm_time_since_inj_diff_sqsqrt',

    'lm_hbi_diff',
    'lm_hbi_diff_sqsqrt',

    'nlm_hbi_diff',
    'nlm_hbi_diff_sqsqrt'
]

confidence_intervals = {}
for var_name in reduced_columns: 
    confidence_intervals[f'{var_name}_upper'] = np.nanmean(final_performance[var_name]) + (1.96 * (np.std(final_performance[var_name]) / (np.sqrt(sum(~np.isnan(final_performance[var_name]))) )) )
    confidence_intervals[f'{var_name}_lower'] = np.nanmean(final_performance[var_name]) - (1.96 * (np.std(final_performance[var_name]) / (np.sqrt(sum(~np.isnan(final_performance[var_name]))) )) )
    confidence_intervals[f'{var_name}_mean'] = np.nanmean(final_performance[var_name])
    print(f"{var_name}")
    print(f"    {confidence_intervals[f'{var_name}_mean']:.2f} [{confidence_intervals[f'{var_name}_upper']:.2f}, {confidence_intervals[f'{var_name}_lower']:.2f}]\n")

print(f"Like-Me size min: {final_performance['like_me_value'].min()}, Max: {final_performance['like_me_value'].max()}")

# p-vals
p_value_pairs = [
    ('lm_mean_euclid_dist_to_pt', 'nlm_mean_euclid_dist_to_pt'),

    ('lm_mean_subj_diff_sx', 'nlm_mean_subj_diff_sx'),
    ('lm_mean_subj_diff_sx_sqsqrt', 'nlm_mean_subj_diff_sx_sqsqrt'),

    ('lm_mean_subj_diff_rtp', 'nlm_mean_subj_diff_rtp'),
    ('lm_mean_subj_diff_rtp_sqsqrt', 'nlm_mean_subj_diff_rtp_sqsqrt'),

    ('lm_age_diff', 'nlm_age_diff'),
    ('lm_age_diff_sqsqrt', 'nlm_age_diff_sqsqrt'),

    ('lm_time_since_inj_diff', 'nlm_time_since_inj_diff'),
    ('lm_time_since_inj_diff_sqsqrt', 'nlm_time_since_inj_diff_sqsqrt'),

    ('lm_hbi_diff', 'nlm_hbi_diff'),
    ('lm_hbi_diff_sqsqrt', 'nlm_hbi_diff_sqsqrt')
]

for lm_col, nlm_col in p_value_pairs:
    lm = final_performance[lm_col]
    nlm = final_performance[nlm_col]

    keep_idx = ~np.isnan(lm) & ~np.isnan(nlm)
    lm_clean = lm[keep_idx]
    nlm_clean = nlm[keep_idx]

    t_stat, p_val = stats.ttest_rel(lm_clean, nlm_clean)

    print(lm_clean.head())
    print(f"t = {t_stat:.3f}, p = {p_val:.10f}\n")

###############
## plots
# hist pt_mahalanobis_distance_from_cohort
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(final_performance['pt_mahalanobis_distance_from_cohort'], bins='auto', density=True, alpha=0.4, edgecolor='white', color='grey')
sns.kdeplot(x=final_performance['pt_mahalanobis_distance_from_cohort'], ax=ax, fill=False, alpha=0.9, color='dimgrey', linewidth=1.8)
ax.set_xlabel("Mahalanobis Distance from Cohort")
ax.set_ylabel("Density")
plt.tight_layout()
plt.show()

# hist like_me_value
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(final_performance['like_me_value'], bins=20, density=True, alpha=0.4, edgecolor='white', color='grey', label='Histogram')
sns.kdeplot(x=final_performance['like_me_value'], ax=ax, fill=False, alpha=0.9, color='dimgrey', linewidth=1.8, label='Kernel Density Estimation')
ax.set_xlabel("Size of Like-Me Sub-Cohort (n)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
plt.tight_layout()
plt.savefig("./figs/performance_like_me_values.png", dpi=300)
plt.show()

#################
## Euclid Dist 
#################
# hist lm_mean_euclid_dist_to_pt and nlm_mean_euclid_dist_to_pt
fig, ax = plt.subplots(figsize=(8,4))
ax.hist(final_performance['lm_mean_euclid_dist_to_pt'], bins=20, density=True, alpha=0.6, edgecolor='white', color='slategrey', label='Like-Me Sub-Cohort')
ax.hist(final_performance['nlm_mean_euclid_dist_to_pt'], bins=40, density=True, alpha=0.5, edgecolor='white', color='dimgrey', label='All other reference subjects')
sns.kdeplot(x=final_performance['lm_mean_euclid_dist_to_pt'], ax=ax, fill=False, alpha=0.8, color='slategrey', linewidth=1.8)
sns.kdeplot(x=final_performance['nlm_mean_euclid_dist_to_pt'], ax=ax, fill=False, alpha=0.8, color='dimgrey', linewidth=1.8)
ax.set_xlabel("Euclidean Distance to Patient (similarity confirmation)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/performance_euclidean_distances.png", dpi=300)
plt.show()

################
## demographics
################
# abs age
fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(x=(final_performance['lm_age_diff_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='slategrey', linewidth=2, label='Like-Me Sub-Cohort')
sns.kdeplot(x=(final_performance['nlm_age_diff_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='grey', linewidth=2, label='All other reference subjects')
plt.axvline(x=0, ymax=0.95, color='lightgrey', linestyle='--', linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Absolute difference between patient and group mean age (years)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/performance_age.png", dpi=300)
plt.show()

# abs time since inj
fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(x=(final_performance['lm_time_since_inj_diff_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='slategrey', linewidth=2, label='Like-Me Sub-Cohort')
sns.kdeplot(x=(final_performance['nlm_time_since_inj_diff_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='grey', linewidth=2, label='All other reference subjects')
plt.axvline(x=0, ymax=0.95, color='lightgrey', linestyle='--', linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Absolute difference between patient and group mean time since injury (days)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/performance_time_since_injury.png", dpi=300)
plt.show()

# abs hbi score
fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(x=(final_performance['lm_hbi_diff_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='slategrey', linewidth=2, label='Like-Me Sub-Cohort')
sns.kdeplot(x=(final_performance['nlm_hbi_diff_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='grey', linewidth=2, label='All other reference subjects')
plt.axvline(x=0, ymax=0.95, color='lightgrey', linestyle='--', linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Absolute difference between patient and group mean symptom severity (HBI score)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/performance_hbi.png", dpi=300)
plt.show()

################
## SX time
################
# raw time sx diffs
fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(x=(final_performance['lm_mean_subj_diff_sx']*-1), ax=ax, fill=True, alpha=0.4, color='slategrey', linewidth=2, label='Like-Me Sub-Cohort')
sns.kdeplot(x=(final_performance['nlm_mean_subj_diff_sx']*-1), ax=ax, fill=True, alpha=0.4, color='grey', linewidth=2, label='All other reference subjects')
plt.axvline(x=0, ymax=0.95, color='lightgrey', linestyle='--', linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Difference in time to symptom resolution (patient - group mean, days)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/raw_time_sx_diff.png", dpi=300)
plt.show()

################
## RTP
################
# raw rtp diffs
fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(x=(final_performance['lm_mean_subj_diff_rtp']*-1), ax=ax, fill=True, alpha=0.4, color='slategrey', linewidth=2, label='Like-Me Sub-Cohort')
sns.kdeplot(x=(final_performance['nlm_mean_subj_diff_rtp']*-1), ax=ax, fill=True, alpha=0.4, color='grey', linewidth=2, label='All other reference subjects')
plt.axvline(x=0, ymax=0.95, color='lightgrey', linestyle='--', linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Difference in time to RTP (patient - group mean, days)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/raw_time_rtp_diff.png", dpi=300)
plt.show()

###########
## Absolute value
###########
################
## SX time
################
# Absolute value of differences for time sx
fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(x=(final_performance['lm_mean_subj_diff_sx_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='slategrey', linewidth=2, label='Like-Me Sub-Cohort')
sns.kdeplot(x=(final_performance['nlm_mean_subj_diff_sx_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='grey', linewidth=2, label='All other reference subjects')
plt.axvline(x=0, ymax=0.95, color='lightgrey', linestyle='--', linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Absolute difference between patient and group mean time to symptom resolution (days)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/abs_time_sx_diff.png", dpi=300)
plt.show()

################
## RTP
################
# Absolute value of differences for RTP
fig, ax = plt.subplots(figsize=(8,4))
sns.kdeplot(x=(final_performance['lm_mean_subj_diff_rtp_sqsqrt']), ax=ax, fill=True, alpha=0.4, color='slategrey', linewidth=2, label='Like-Me Sub-Cohort')
sns.kdeplot(x=(final_performance['nlm_mean_subj_diff_rtp_sqsqrt']), ax=ax, fill=True, alpha=0.3, color='gray', linewidth=2, label='All other reference subjects')
plt.axvline(x=0, ymax=0.95, color='lightgrey', linestyle='--', linewidth=1.5, label="Perfect Prediction")
ax.set_xlabel("Absolute difference between patient and group mean time to RTP (days)")
ax.set_ylabel("Density")
ax.set_yticks([]) 
ax.legend(frameon=True, loc='upper right')
plt.tight_layout()
plt.savefig("./figs/abs_time_rtp_diff.png", dpi=300)
plt.show()