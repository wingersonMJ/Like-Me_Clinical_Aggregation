from distances import cohort, y, patient, patient_y
from distances import like_me_cohort, like_me_y
from distances import like_me_cohort_original, patient_original
from distances import sample, idx

import pandas as pd 
import numpy as np
import time as time

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
start = time.time()

randomness = [42, 43, 44, 45, 1997, 1989]
neighbors = [5, 10, 20, 40, 80, 100, 150, 200]
min_d = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]
metric = ['euclidean', 'cosine', 'correlation', 'mahalanobis', 'yule']
spread = [0.001, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 4.0]

umap_performance = []

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
                    umap_performance.append({
                        'randomness': r,
                        'neighbors': n,
                        'min_d': d,
                        'metric': m,
                        'spread': s,
                        'within_sum_of_squares': within_ss,
                        'between_sum_of_squares': between_ss,
                        'ratio': ratio
                    })

end = time.time()
print(f"{(end-start)/60} minutes to run")

# save performance
umap_performance = pd.DataFrame(umap_performance)
umap_performance.sort_values(by='ratio', ascending=True, inplace=True)
umap_performance.to_csv("../Data/umap_performance.csv")

umap_performance.head()

#############
# final UMAP with tuned hyper_params
reducer = umap.UMAP(
    n_neighbors=umap_performance['neighbors'].iloc[0],
    min_dist=umap_performance['min_d'].iloc[0],
    metric=umap_performance['metric'].iloc[0],
    spread=umap_performance['spread'].iloc[0],
    random_state=umap_performance['randomness'].iloc[0]
    )
two_dim = reducer.fit_transform(cohort)

# plot
plt.figure(figsize=(6, 5))
plt.scatter(two_dim[cohort_mask, 0], two_dim[cohort_mask, 1], s=16, alpha=0.7, label='Cohort', color='lightgrey')
plt.scatter(two_dim[like_me_mask, 0], two_dim[like_me_mask, 1], s=28, alpha=0.9, label='Like-me patients', color='slategrey')
plt.scatter(two_dim[patient_mask, 0], two_dim[patient_mask, 1], s=80, marker='x', linewidths=2, label='Patient', color='firebrick')
plt.legend(loc='best', frameon=False)
plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2')
plt.tight_layout()
plt.savefig("./figs/UMAP", dpi=300)
plt.show()

##############
# try t-sne
from sklearn.manifold import TSNE

t_reducer = TSNE(n_components = 2, random_state=1989)
tsne_data = t_reducer.fit_transform(cohort)

# plot
plt.figure(figsize=(6, 5))
plt.scatter(tsne_data[cohort_mask, 0], tsne_data[cohort_mask, 1], s=16, alpha=0.7, label='Cohort', color='lightgrey')
plt.scatter(tsne_data[like_me_mask, 0], tsne_data[like_me_mask, 1], s=28, alpha=0.9, label='Like-me patients', color='slategrey')
plt.scatter(tsne_data[patient_mask, 0], tsne_data[patient_mask, 1], s=80, marker='x', linewidths=2, label='Patient', color='firebrick')
plt.legend(loc='best', frameon=False)
plt.xlabel('TSNE-1'); plt.ylabel('TSNE-2')
plt.tight_layout()
plt.savefig("./figs/TSNE", dpi=300)
plt.show()
