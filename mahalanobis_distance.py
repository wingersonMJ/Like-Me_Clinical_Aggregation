from pre_processing import X
from pre_processing import y

import random 

import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt
import seaborn as sn
from pywaffle import Waffle

# assign X from pre_processing to df
cohort = X
cohort.head()

# randomly select a row from cohort to use as example patient
random.seed(12)
sample = random.randint(0, 557)
print(sample)

patient = X.iloc[sample,:]
print(patient)

patient_y = y.iloc[sample]
print(patient_y)

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

pt_mahalanobis_distance_from_cohort = mahalanobis(x=patient, data=cohort)
print(pt_mahalanobis_distance_from_cohort)

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

###################
## Plot KDE of mahalanobis_distance_to_cohort
# plot cohort, vertical line for pt_mahalanobis_distance_from_cohort
cohort['mean_centered_mahalanobis_distance_to_cohort'] = cohort['mahalanobis_distance_to_cohort'] - np.mean(cohort['mahalanobis_distance_to_cohort'])
mean_centered_pt_mahalanobis_distance_from_cohort = pt_mahalanobis_distance_from_cohort - np.mean(cohort['mahalanobis_distance_to_cohort'])

plt.figure()
sn.kdeplot(
    x=cohort['mean_centered_mahalanobis_distance_to_cohort'],
    fill=True,
    alpha=0.5,
    color='steelblue')
plt.axvline(mean_centered_pt_mahalanobis_distance_from_cohort, color='firebrick', linestyle='--', linewidth=2, label="Patient Mahalanobis Distance from Cohort")
plt.xlabel("Mahalanobis Distances")
plt.ylabel("Smoothed Kernel Density")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1))
plt.savefig("./figs/kde_plot_example", dpi=300)
plt.show()

#################
## like-me cohort
like_me_value = 40
like_me_by = 'mahalanobis_distance_to_patient'
idx = cohort[like_me_by].nsmallest(like_me_value).index
like_me_cohort = cohort.loc[idx]
like_me_y = y.loc[idx]

like_me_cohort.head()
like_me_y.head()

###########
## Undo min-max scaling

##############
## Summary stats 
like_me_cohort.describe()
like_me_y.describe()

print(patient)
print(patient_y)

#################
# box plots for some vars 
# time_since_injury
fig, axes = plt.subplots(2, 2, figsize=(10, 7))

sn.boxplot(data=like_me_cohort['time_since_injury'], ax=axes[0, 0], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort['time_since_injury'], ax=axes[0, 0], color='dimgrey', size=4, jitter=True)
axes[0, 0].scatter(0, patient['time_since_injury'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5, label='Patient value')
axes[0, 0].set_ylabel("Time since injury (days)", fontsize=12)
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=12)

sn.boxplot(data=like_me_cohort['age'], ax=axes[0, 1], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort['age'], ax=axes[0, 1], color='dimgrey', size=4, jitter=True)
axes[0, 1].scatter(0, patient['age'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[0, 1].set_ylabel("Age (years)", fontsize=12)

sn.boxplot(data=like_me_cohort['HBI_total'], ax=axes[1, 0], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort['HBI_total'], ax=axes[1, 0], color='dimgrey', size=4, jitter=True)
axes[1, 0].scatter(0, patient['HBI_total'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[1, 0].set_ylabel("Symptom burden (HBI total score)", fontsize=12)

sn.boxplot(data=like_me_cohort['headache_severity'], ax=axes[1, 1], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort['headache_severity'], ax=axes[1, 1], color='dimgrey', size=4, jitter=True)
axes[1, 1].scatter(0, patient['headache_severity'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[1, 1].set_ylabel("Headache severity", fontsize=12)

plt.tight_layout()
plt.savefig("./figs/like_me_aggregated_X", dpi=300, bbox_inches='tight')
plt.show()

################
## plot y as well
# time to sx res
plt.figure(figsize=(12,8))
sn.kdeplot(
    x=like_me_y['time_sx'],
    fill=True,
    alpha=0.5,
    color='steelblue')
plt.axvline(patient_y['time_sx'], color='firebrick', linestyle='--', linewidth=2, label=f"Patient Time to Symptom Resolution: {patient_y['time_sx']:.0f} days")
plt.axvline(np.nanmean(like_me_y['time_sx']), color='lightgrey', linestyle='--', linewidth=2, label=f"Mean Like-Me-Cohort Time to Symptom Resolution: {np.nanmean(like_me_y['time_sx']):.1f} days")
plt.axvline(np.nanmedian(like_me_y['time_sx']), color='lightgrey', linestyle=':', linewidth=2, label=f"Median Like-Me-Cohort Time to Symptom Resolution: {np.nanmedian(like_me_y['time_sx']):.1f} days")
plt.xlabel("Time to Symptom Resolution (days)")
plt.ylabel("Smoothed Kernel Density")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.125))
plt.tight_layout()
plt.savefig("./figs/like_me_aggregated_time_sx", dpi=300)
plt.show()

# time to rtp
plt.figure(figsize=(12,8))
sn.kdeplot(
    x=like_me_y['time_rtp'],
    fill=True,
    alpha=0.5,
    color='steelblue')
plt.axvline(patient_y['time_rtp'], color='firebrick', linestyle='--', linewidth=2, label=f"Patient Time to Return-to-Play: {patient_y['time_rtp']:.0f} days")
plt.axvline(np.nanmean(like_me_y['time_rtp']), color='lightgrey', linestyle='--', linewidth=2, label=f"Mean Like-Me-Cohort Time to Return-to-Play: {np.nanmean(like_me_y['time_rtp']):.1f} days")
plt.axvline(np.nanmedian(like_me_y['time_rtp']), color='lightgrey', linestyle=':', linewidth=2, label=f"Median Like-Me-Cohort Time to Return-to-Play: {np.nanmedian(like_me_y['time_rtp']):.1f} days")
plt.xlabel("Time to Return-to-Play (days)")
plt.ylabel("Smoothed Kernel Density")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.125))
plt.tight_layout()
plt.savefig("./figs/like_me_aggregated_time_rtp", dpi=300)
plt.show()

# waffle plot for percent psac
nanFilled_PPCS = pd.to_numeric(like_me_y['PPCS'], errors='coerce')
cats = nanFilled_PPCS.map({0: 'No PSaC', 1: 'PSaC'}).fillna('Missing')
order = ['No PSaC', 'PSaC', 'Missing']
values = {k: cats.value_counts().get(k, 0) for k in order}
colors = ['#2ca02c', '#d62728', '#bdbdbd']

plt.figure(
    figsize=(12,8),
    FigureClass=Waffle,
    rows=5,
    values=values,
    colors=colors,
    legend={'labels': [f"{k}: {values[k]}" for k in order], 'loc': 'upper right', 'fontsize': 14},
    title={'label': "PSaC development in Like-Me Cohort", 'loc': 'center', 'fontsize': 16, 'fontweight': 'bold'},
    block_arranging_style='snake'
)
plt.tight_layout()
plt.savefig("./figs/like_me_aggregated_PSaC", dpi=300)
plt.show()