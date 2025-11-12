from pre_processing import X, y
from pre_processing import X_mins, X_ranges
from pre_processing import categorical_scaler, categoricals

import random 
import numpy as np
import pandas as pd
import time
import datetime as datetime
pd.set_option("display.max_columns", None)

from tableone import TableOne

import matplotlib.pyplot as plt

# get cohort
"X.head()
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
like_me_performance = pd.DataFrame(columns=['pt_mahalanobis_distance_from_cohort', 'mean_euclid_dist_to_pt', 'like_me_value', 'lm_subj_diff_sx', 'lm_mean_subj_diff_sx', 'lm_median_subj_diff_sx', 'nlm_subj_diff_sx', 'nlm_mean_subj_diff_sx', 'nlm_median_subj_diff_sx'])

start = time.time()
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
    mean_euclid_dist_to_pt = cohort['euclidean_distance'].mean()

    ###########################
    # Get like-me cohort 
    like_me_value = (10 - pt_mahalanobis_distance_from_cohort)*10
    like_me_value = np.round(like_me_value, decimals=0)
    like_me_by = 'euclidean_distance'
    like_me_idx = cohort[like_me_by].nsmallest(like_me_value).index
    like_me_cohort = cohort.loc[like_me_idx]
    like_me_y = y.loc[like_me_idx]

    not_like_me_cohort = cohort.loc[~like_me_idx]
    not_like_me_y = y.loc[~like_me_idx]

    ##########################
    # Get differences between the patient and the like-me cohort in y's
    # Mean difference between the patient y and the like-me cohort individual values
    lm_subj_diff_sx = like_me_y['time_sx'] - patient_y['time_sx']
    lm_subj_diff_sx = lm_subj_diff_sx.mean()

    # Difference between the patient y and the mean of the like-me cohort values
    lm_mean_subj_diff_sx = like_me_y['time_sx'].mean() - patient_y['time_sx']
    
    # Difference between the patient y and the median of the like-me cohort values
    lm_median_subj_diff_sx = like_me_y['time_sx'].median() - patient_y['time_sx']

    ##########################
    # Get differences between the patient and the NOT-like-me cohort in y's
    # Mean difference between the patient y and the like-me cohort individual values
    nlm_subj_diff_sx = not_like_me_y['time_sx'] - patient_y['time_sx']
    nlm_subj_diff_sx = lm_subj_diff_sx.mean()

    # Difference between the patient y and the mean of the like-me cohort values
    nlm_mean_subj_diff_sx = not_like_me_y['time_sx'].mean() - patient_y['time_sx']
    
    # Difference between the patient y and the median of the like-me cohort values
    nlm_median_subj_diff_sx = not_like_me_y['time_sx'].median() - patient_y['time_sx']

    # get the important metrics
    iteration_performance = {
        'pt_mahalanobis_distance_from_cohort': pt_mahalanobis_distance_from_cohort, 
        'mean_euclid_dist_to_pt': mean_euclid_dist_to_pt, 
        'like_me_value': like_me_value, 
        'lm_subj_diff_sx': lm_subj_diff_sx, 
        'lm_mean_subj_diff_sx': lm_mean_subj_diff_sx, 
        'lm_median_subj_diff_sx': lm_median_subj_diff_sx, 
        'nlm_subj_diff_sx': nlm_subj_diff_sx, 
        'nlm_mean_subj_diff_sx': nlm_mean_subj_diff_sx, 
        'nlm_median_subj_diff_sx': nlm_median_subj_diff_sx
    }

    # save
    like_me_performance.loc[len(like_me_performance)] = iteration_performance
    current_time = datetime.now().strftime("%Y-%m-%d")
    save_name = f"../Data/like_me_performance{current_time}.csv"
    like_me_performance.to_csv(save_name)
end = time.time()
print(f"{(end-start)/60} minutes to run")


################
# load back in the nresults
final_performance = pd.read_csv(save_name) # replace with actual file name: "../Data/like_me_performance{date}.csv"

final_performance.head()

###############
# summary stats
columns=['pt_mahalanobis_distance_from_cohort', 'mean_euclid_dist_to_pt', 'like_me_value', 'lm_subj_diff_sx', 'lm_mean_subj_diff_sx', 'lm_median_subj_diff_sx', 'nlm_subj_diff_sx', 'nlm_mean_subj_diff_sx', 'nlm_median_subj_diff_sx']
mytable = TableOne(final_performance, columns=columns, continuous=columns, pval=True)
print(mytable.tabulate(tablefmt = "fancy_grid"))

###############
# plots