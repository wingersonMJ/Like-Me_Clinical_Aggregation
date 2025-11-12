from distances import cohort, y, patient, patient_y
from distances import like_me_cohort, like_me_y
from distances import like_me_cohort_original, patient_original
from distances import sample, idx

import pandas as pd 
import numpy as np
import time as time
from datetime import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn

import umap

###########
cohort.head()
print(patient)

# create masks for each subject
like_me_mask = np.zeros(len(cohort), dtype=bool)
like_me_mask[np.array(idx, dtype=int)] = True

patient_mask = np.zeros(len(cohort), dtype=bool)
patient_mask[sample] = True

cohort_mask = ~(patient_mask | like_me_mask)

################
# UMAP for loop to tune hyper_params
randomness = [42, 43, 1997, 1989] #
neighbors = [10, 20, 40, 80, 100, 150, 200] #
min_d = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50] #
metric = ['euclidean', 'cosine', 'correlation', 'mahalanobis'] #
spread = [0.10, 0.20, 0.30, 0.60, 0.80, 0.99] #

umap_performance = pd.DataFrame(columns=['randomness', 'neighbors', 'min_d', 'metric', 'spread', 'within_sum_of_squares', 'between_sum_of_squares', 'ratio'])

number_of_combinations = (len(randomness)*len(neighbors)*len(min_d)*len(metric)*len(spread))
print(number_of_combinations)

start = time.time()
for r in randomness:
    for n in neighbors:
        for d in min_d:
            for m in metric:
                for s in spread:

                    # spread must be >= min_d
                    if s < d:
                        continue

                    # umap process
                    reducer = umap.UMAP(
                        n_neighbors=n,
                        min_dist=d,
                        metric=m,
                        spread=s,
                        random_state=r
                        )
                    two_dim = reducer.fit_transform(cohort)

                    # separate idx for within like-me sample and not
                    within_idx = np.where(like_me_mask)[0]
                    between_idx = np.where(cohort_mask)[0]

                    # calculate centroids as the patient point
                    patient_point = two_dim[patient_mask][0]

                    within_centroid = patient_point
                    between_centroid = patient_point

                    # within euclidean distance sum of squares
                    within_ss = 0
                    for i in within_idx:
                        # calc euclidean distance
                        euclid_dist_within = np.linalg.norm(two_dim[i] - within_centroid)
                        within_ss += np.dot(euclid_dist_within, euclid_dist_within)
            
                    # between sum of squares
                    between_ss = 0
                    for j in between_idx:
                        # calc euclidean distance
                        euclid_dist_btwn = np.linalg.norm(two_dim[j] - between_centroid)
                        between_ss += np.dot(euclid_dist_btwn, euclid_dist_btwn)
                    
                    if between_ss == 0:
                        between_ss += 0.0001
                    # get the ratio
                    ratio = within_ss/between_ss

                    # save params and performance
                    iteration_performance = {
                        'randomness': r,
                        'neighbors': n,
                        'min_d': d,
                        'metric': m,
                        'spread': s,
                        'within_sum_of_squares': within_ss,
                        'between_sum_of_squares': between_ss,
                        'ratio': ratio
                    }
                    umap_performance.loc[len(umap_performance)] = iteration_performance

                    current_time = datetime.now().strftime("%Y-%m-%d")
                    save_name = f"../Data/umap_performance_final.csv"
                    umap_performance.to_csv(save_name)
end = time.time()
print(f"{(end-start)/60} minutes to run")

###################################
# load final doc and sort performance
final_performance = pd.read_csv(save_name) # replace with actual file name: "../Data/umap_performance_{date}.csv"
final_performance.sort_values(by='ratio', ascending=True, inplace=True)

final_performance.head()

#############
# final UMAP with tuned hyper_params
reducer = umap.UMAP(
    n_neighbors=final_performance['neighbors'].iloc[0],
    min_dist=final_performance['min_d'].iloc[0],
    metric=final_performance['metric'].iloc[0],
    spread=final_performance['spread'].iloc[0],
    random_state=final_performance['randomness'].iloc[0]
    )
two_dim = reducer.fit_transform(cohort)

# plot
plt.figure(figsize=(6, 5))
plt.scatter(two_dim[cohort_mask, 0], two_dim[cohort_mask, 1], s=16, alpha=0.7, label='Cohort', color='lightgrey')
plt.scatter(two_dim[like_me_mask, 0], two_dim[like_me_mask, 1], s=28, alpha=0.9, label='Like-Me Sub-Cohort', color='slategrey')
plt.scatter(two_dim[patient_mask, 0], two_dim[patient_mask, 1], s=80, marker='x', linewidths=2, label='Example Patient', color='firebrick')
plt.legend(loc='best', frameon=False)
plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
plt.tight_layout()
plt.savefig("./figs/UMAP_1", dpi=300)

# tuned with second best UMAP params
reducer = umap.UMAP(
    n_neighbors=final_performance['neighbors'].iloc[1],
    min_dist=final_performance['min_d'].iloc[1],
    metric=final_performance['metric'].iloc[1],
    spread=final_performance['spread'].iloc[1],
    random_state=final_performance['randomness'].iloc[1]
    )
two_dim = reducer.fit_transform(cohort)

# plot
plt.figure(figsize=(6, 5))
plt.scatter(two_dim[cohort_mask, 0], two_dim[cohort_mask, 1], s=16, alpha=0.7, label='Cohort', color='lightgrey')
plt.scatter(two_dim[like_me_mask, 0], two_dim[like_me_mask, 1], s=28, alpha=0.9, label='Like-Me Sub-Cohort', color='slategrey')
plt.scatter(two_dim[patient_mask, 0], two_dim[patient_mask, 1], s=80, marker='x', linewidths=2, label='Example Patient', color='firebrick')
plt.legend(loc='best', frameon=False)
plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
plt.tight_layout()
plt.savefig("./figs/UMAP_2", dpi=300)