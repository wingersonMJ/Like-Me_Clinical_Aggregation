from pre_processing import X
import random 
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

# assign X from pre_processing to df
cohort = X
cohort.head()

# randomly select a row from cohort to use as example patient
random.seed(1989)
sample = random.randint(0, 557)
print(sample)

patient = X.iloc[sample,:]
print(patient)

cohort = cohort.drop(index=cohort.index[sample])

###############
# patient mah distance from cohort
def mahalanobis(x=None, data=None, cov=None):
    x_minus_mu = x - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    return np.sqrt(mahal)

mahalanobis_distance_from_cohort = mahalanobis(x=patient, data=cohort)
print(mahalanobis_distance_from_cohort)

###################
# each cohort subject's mah distance
# set up cov matrix
cov = np.cov(cohort.values.T)
inv_covmat = np.linalg.inv(cov)
cohort_mean = np.mean(cohort)

# compare patient to everyone in the cohort
for i, row in cohort.iterrows():
    
    # each cohort subject's mah dist from cohort
    x_minus_mu = row - cohort_mean
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    mah_dist_to_cohort = np.sqrt(mahal)

    # patients mah distance from each subject
    x_mu = patient - row
    left = np.dot(x_mu, inv_covmat)
    mahal = np.dot(left, x_mu.T)
    mah_dist_to_pt = np.sqrt(mahal)

    # cosine similarity
    mult_norm = (np.linalg.norm(row) * np.linalg.norm(patient))
    cosine_similarity = (np.dot(row, patient)) / mult_norm

    # normalized dot prod
    max_norm = max(np.linalg.norm(row), np.linalg.norm(patient))
    norm_dot_prod = (np.dot(row, patient)) / max_norm

    # euclidean distance
    euclid_dist = np.linalg.norm(row - patient)

    # add them all to the data frame
    cohort.loc[i, "mahalanobis_distance_to_cohort"] = mah_dist_to_cohort
    cohort.loc[i, "mahalanobis_distance_to_patient"] = mah_dist_to_pt
    cohort.loc[i, "cosine_similarity"] = cosine_similarity
    cohort.loc[i, "normalized_dot_product_distance"] = norm_dot_prod
    cohort.loc[i, "euclidean_distance"] = euclid_dist

metrics = ['mahalanobis_distance_to_cohort', 'mahalanobis_distance_to_patient', 'cosine_similarity', 'normalized_dot_product_distance', 'euclidean_distance']
cohort[metrics].head()

