from pre_processing import X, y
from pre_processing import X_mins, X_ranges
from pre_processing import categorical_scaler, categoricals

import random 
import numpy as np
import pandas as pd
import time
from datetime import datetime
pd.set_option("display.max_columns", None)

from tableone import TableOne

import matplotlib.pyplot as plt

# get cohort
X.head()
X.describe()

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
        'nlm_median_subj_diff_rtp' 
    ])

start = time.time()
iteration = 0
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
        euclid_dist = np.linalg.norm(row - patient) 
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

    ##########################
    # time sx res
    ########################
    # Difference between the patient y and the mean of the like-me cohort values
    lm_mean_subj_diff_sx = np.nanmean(like_me_y['time_sx']) - patient_y['time_sx']
    
    # Difference between the patient y and the median of the like-me cohort values
    lm_median_subj_diff_sx = np.nanmedian(like_me_y['time_sx']) - patient_y['time_sx']

    # euclid dist to pt
    for i, row in like_me_cohort.iterrows():
        like_me_cohort['lm_euclid_dist'] = np.linalg.norm(row - patient) 
    lm_mean_euclid_dist_to_pt = np.nanmean(like_me_cohort['lm_euclid_dist'])

    ##########################
    # Difference between the patient y and the mean of the like-me cohort values
    nlm_mean_subj_diff_sx = np.nanmean(not_like_me_y['time_sx']) - patient_y['time_sx']
    
    # Difference between the patient y and the median of the like-me cohort values
    nlm_median_subj_diff_sx = np.nanmedian(not_like_me_y['time_sx']) - patient_y['time_sx']

    # euclid dist to pt
    for i, row in not_like_me_cohort.iterrows():
        not_like_me_cohort['nlm_euclid_dist'] = np.linalg.norm(row - patient) 
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
        'nlm_median_subj_diff_rtp': nlm_median_subj_diff_rtp 
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

final_performance.columns

#################
# add squared diff
squared_cols = ['lm_mean_subj_diff_sx',  'nlm_mean_subj_diff_sx', 'lm_median_subj_diff_sx', 'nlm_median_subj_diff_sx', 'lm_mean_subj_diff_rtp',  'nlm_mean_subj_diff_rtp', 'lm_median_subj_diff_rtp', 'nlm_median_subj_diff_rtp']
for col in squared_cols:
    final_performance[f"{col}_sq"] = (final_performance[col])**2
    final_performance[f"{col}_sqsqrt"] = np.sqrt(final_performance[f"{col}_sq"])

###############
# summary stats
columns=['like_me_value', 'pt_mahalanobis_distance_from_cohort', 'mean_euclid_dist_to_pt', 'lm_mean_euclid_dist_to_pt', 'nlm_mean_euclid_dist_to_pt', 'lm_mean_subj_diff_sx', 'nlm_mean_subj_diff_sx', 'lm_median_subj_diff_sx', 'nlm_median_subj_diff_sx', 'lm_mean_subj_diff_sx_sq',  'nlm_mean_subj_diff_sx_sq', 'lm_median_subj_diff_sx_sq', 'nlm_median_subj_diff_sx_sq', 'lm_mean_subj_diff_sx_sqsqrt',  'nlm_mean_subj_diff_sx_sqsqrt', 'lm_median_subj_diff_sx_sqsqrt', 'nlm_median_subj_diff_sx_sqsqrt', 'lm_mean_subj_diff_rtp', 'nlm_mean_subj_diff_rtp', 'lm_median_subj_diff_rtp', 'nlm_median_subj_diff_rtp', 'lm_mean_subj_diff_rtp_sq',  'nlm_mean_subj_diff_rtp_sq', 'lm_median_subj_diff_rtp_sq', 'nlm_median_subj_diff_rtp_sq', 'lm_mean_subj_diff_rtp_sqsqrt',  'nlm_mean_subj_diff_rtp_sqsqrt', 'lm_median_subj_diff_rtp_sqsqrt', 'nlm_median_subj_diff_rtp_sqsqrt']

mytable = TableOne(final_performance, columns=columns, continuous=columns, pval=False)
print(mytable.tabulate(tablefmt = "github"))

# means and 95% CI's
confidence_intervals = {}
for var_name in columns: 
    confidence_intervals[f'{var_name}_upper'] = np.nanmean(final_performance[var_name]) + (1.96 * (np.std(final_performance[var_name]) / (np.sqrt(sum(~np.isnan(final_performance[var_name]))) )) )
    confidence_intervals[f'{var_name}_lower'] = np.nanmean(final_performance[var_name]) - (1.96 * (np.std(final_performance[var_name]) / (np.sqrt(sum(~np.isnan(final_performance[var_name]))) )) )
    confidence_intervals[f'{var_name}_mean'] = np.nanmean(final_performance[var_name])
    print(f"{var_name}")
    print(f"    {confidence_intervals[f'{var_name}_mean']:.2f} [{confidence_intervals[f'{var_name}_upper']:.2f}, {confidence_intervals[f'{var_name}_lower']:.2f}]\n")

###############
# plots
