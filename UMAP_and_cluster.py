from mahalanobis_distance import cohort, y, patient, patient_y
from mahalanobis_distance import like_me_cohort, like_me_y
from mahalanobis_distance import like_me_cohort_original, patient_original
from mahalanobis_distance import sample, idx

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn

import umap

###########
# UMAP or other visualization for the whole cohort, the 40, and the one pt
cohort.head()

reducer = umap.UMAP()
two_dim = reducer.fit_transform(cohort)

# create masks for each subject
like_me_mask = np.zeros(len(cohort), dtype=bool)
like_me_mask[np.array(idx, dtype=int)] = True

patient_mask = np.zeros(len(cohort), dtype=bool)
patient_mask[sample] = True

cohort_mask = ~(patient_mask | like_me_mask)

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




# Do clustering algorithm


# Visualize clustering alg 