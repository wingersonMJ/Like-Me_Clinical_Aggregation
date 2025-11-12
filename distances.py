from pre_processing import X, y
from pre_processing import X_mins, X_ranges
from pre_processing import categorical_scaler, categoricals

import random 
import numpy as np
import pandas as pd
pd.set_option("display.max_columns", None)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sn
from pywaffle import Waffle

# assign X from pre_processing to df
cohort = X
cohort.head()
cohort.describe()

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

metrics = ['mahalanobis_distance_to_patient', 'cosine_similarity', 'normalized_dot_product_distance', 'euclidean_distance']
cohort[metrics].head()
print(cohort['euclidean_distance'].min(), cohort['euclidean_distance'].max())

###############
## Quick comparisons between metrics
sn.pairplot(data=cohort[metrics])
plt.savefig("./figs/metrics_plot")

###################
## Plot KDE of mahalanobis_distance_to_cohort
# plot cohort, vertical line for pt_mahalanobis_distance_from_cohort
plt.figure()
sn.kdeplot(
    x=cohort['mahalanobis_distance_to_cohort'],
    fill=True,
    alpha=0.5,
    color='steelblue')
plt.axvline(pt_mahalanobis_distance_from_cohort, color='firebrick', linestyle='--', linewidth=2, label="Patient Mahalanobis Distance from Cohort")
plt.xlabel("Mahalanobis Distances")
plt.ylabel("Smoothed Kernel Density")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1))
plt.savefig("./figs/kde_plot_example", dpi=300)

#################
## like-me cohort
like_me_value = (10 - pt_mahalanobis_distance_from_cohort)*10
like_me_value = np.round(like_me_value, decimals=0).astype(int)
print(like_me_value)
like_me_by = 'euclidean_distance'
# mahalanobis_distance_to_patient
# cosine_similarity or normalized_dot_product_distance - have to switch to max below
# euclidean_distance
idx = cohort[like_me_by].nsmallest(like_me_value).index
like_me_cohort = cohort.loc[idx]
like_me_y = y.loc[idx]

like_me_cohort.head()
like_me_y.head()

###########
## Undo min-max scaling
like_me_cohort_original = (like_me_cohort * X_ranges) + (X_mins)
like_me_cohort_original[categoricals] = (like_me_cohort_original[categoricals])/categorical_scaler

patient_original = (patient * X_ranges) + X_mins
patient_original[categoricals] = (patient_original[categoricals])/categorical_scaler

##############
## Summary stats
# Min-Max values
like_me_cohort.describe()
print(patient)

# Original scale
like_me_cohort_original.describe()
print(patient_original)

# y's - never changed from the original scale
like_me_y.describe()
print(patient_y)

#################
# box plots for some vars 
# time_since_injury
fig, axes = plt.subplots(3, 2, figsize=(12, 10))

sn.boxplot(data=like_me_cohort_original['time_since_injury'], ax=axes[0, 0], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort_original['time_since_injury'], ax=axes[0, 0], color='dimgrey', size=4, jitter=True)
axes[0, 0].scatter(0, patient_original['time_since_injury'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5, label='Patient value')
axes[0, 0].set_ylabel("Time since injury (days)", fontsize=12)
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=12)

sn.boxplot(data=like_me_cohort_original['age'], ax=axes[0, 1], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort_original['age'], ax=axes[0, 1], color='dimgrey', size=4, jitter=True)
axes[0, 1].scatter(0, patient_original['age'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[0, 1].set_ylabel("Age (years)", fontsize=12)

sn.boxplot(data=like_me_cohort_original['HBI_total'], ax=axes[1, 0], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort_original['HBI_total'], ax=axes[1, 0], color='dimgrey', size=4, jitter=True)
axes[1, 0].scatter(0, patient_original['HBI_total'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[1, 0].set_ylabel("Symptom burden (HBI total score)", fontsize=12)

sn.boxplot(data=like_me_cohort_original['headache_severity'], ax=axes[1, 1], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort_original['headache_severity'], ax=axes[1, 1], color='dimgrey', size=4, jitter=True)
axes[1, 1].scatter(0, patient_original['headache_severity'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[1, 1].set_ylabel("Headache severity", fontsize=12)

sn.boxplot(data=like_me_cohort_original['BESS_total'], ax=axes[2, 0], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort_original['BESS_total'], ax=axes[2, 0], color='dimgrey', size=4, jitter=True)
axes[2, 0].scatter(0, patient_original['BESS_total'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[2, 0].set_ylabel("BESS total (all stances)", fontsize=12)

sn.boxplot(data=like_me_cohort_original['number_prior_conc'], ax=axes[2, 1], color='lightgrey', showfliers=False)
sn.stripplot(data=like_me_cohort_original['number_prior_conc'], ax=axes[2, 1], color='dimgrey', size=4, jitter=True)
axes[2, 1].scatter(0, patient_original['number_prior_conc'], s=80, linewidths=2, marker='x', color='firebrick', edgecolor='white', zorder=5)
axes[2, 1].set_ylabel("Number of prior concussions", fontsize=12)

plt.tight_layout()
plt.savefig("./figs/like_me_aggregated_X", dpi=300, bbox_inches='tight')

#############
## plot binary variables too
vars_ = ['current_sleep_problems', 'exercise_since_injury', 'sex1f']
labels = ['Yes', 'No']
colors = ['lightblue', 'slategrey']
counts = {}

for var in vars_:
    counts[var] = [(like_me_cohort_original[var] == 1).sum(), (like_me_cohort_original[var] == 0).sum()]

fig, axes = plt.subplots(3, 1, figsize=(7, 8))

axes[0].pie(counts['sex1f'], labels=None, colors=colors, wedgeprops=dict(linewidth=1, edgecolor="white"))
axes[0].axis('equal'); axes[0].set_title('Biological Sex: Female', fontweight='bold')

axes[1].pie(counts['current_sleep_problems'], labels=None, colors=colors, wedgeprops=dict(linewidth=1, edgecolor="white"))
axes[1].axis('equal'); axes[1].set_title('Current sleep problems', fontweight='bold')

axes[2].pie(counts['exercise_since_injury'], labels=None, colors=colors, wedgeprops=dict(linewidth=1, edgecolor="white"))
axes[2].axis('equal'); axes[2].set_title('Exercising since injury', fontweight='bold')

handles = [
    Patch(facecolor=colors[0], edgecolor="white", label="Yes"),
    Patch(facecolor=colors[1], edgecolor="white", label="No")]
fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05), frameon=False, handlelength=1, prop={'size': 12})
plt.tight_layout()
plt.savefig("./figs/like_me_aggregated_X_categoricals", dpi=300, bbox_inches='tight')

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

# waffle plot for percent psac
nanFilled_PPCS = pd.to_numeric(like_me_y['PPCS'], errors='coerce')
cats = nanFilled_PPCS.map({0: 'No PSaC', 1: 'PSaC'}).fillna('Missing')
order = ['No PSaC', 'PSaC', 'Missing']
values = {k: cats.value_counts().get(k, 0) for k in order}
colors = ['forestgreen', 'firebrick', 'lightgrey']

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