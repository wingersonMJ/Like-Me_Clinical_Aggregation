# A Patients-Like-Me approach to aggregating data in clinical management of concussion

**Purpose:** Complex prediction models, such as neural networks, tree-based classifiers, or even simple regressions, can tell patients and clinicians when an individual is expected to recover from concussion (or a number of other conditions, as these models continue to expand in popularity and use), but suffer from a few main limitations:  
- Most prediction models are trained on large cohorts with diverse patient and injury characteristics
    - Thus, generalization to a more precise patient population in a real-world setting can be limitted - patients are left asking "but do these results apply to *me*?" and clinicians are left wondering if prediction models are accurate in thier patient population  
- Few prediction models can provide clear reasoning for *why* or *how* a prediction is made, or what factors where most influental in the prediction that is made 
    - Clinicians and patients, therefore, may not always strongly consider the predictions made by the model
- Prediction models, especially classifiers, can treat medical diseases as a series of binary decisions made in sequence 
    - Actual clinical management of disease is more complex, and clinicians must use all available information to guide decisions that promote long-term health and recovery, while prediction models often evaluate the best decision to improve the current disease state with little longitudinal context

Taken together, a quality data science and machine learning approach to supporting practitioners might consider:
1. Using similarity matching to take the current individuals's clinical presentation (demographics, injury characteristics, etc) and identify past patients with similar characteristics
2. Explore the health and recovery outcomes of those similar past patients to inform potential recovery for the current individual
3. Package this information together in an easy-to-use dashboard so that clinicians and patients can explore how other's "like them" recovered!

## Methods:

#### An Overview:

1. A sample of patients seen for post-concussion care in our sports medicine clinic (n=558)
    - Limit to just the high level features... this is a proof of concept project, so feasability is most important right now (13 features total)
    - One-hot for categoricals
2. Impute missing data
    - Impution strategy was KNN (K=20) 
    - Works well for numerics, but doesn't have a built-in option for categoricals
        - My desire would be "most frequent" for categorical imputations
        - Without that as a built-in option, KNN already calculates the mean value for the 'K' nearest neighbors
            - Because categoricals are already one-hot encoded, I can just round to the nearest integer to get the "most frequent" for my imputed values
3. Normalize the data, Min-Max
    - Need to do this after the imputation, unfortunately. Otherwise my above rounding option won't work
    - Limitation is that variables are not on the same scale during KNN imputation, which can create some bias. 
4. Mahalanobis distance tells us how similar an individual vector to a distribution of vectors
    - Answers the question of "how unique is the current patient's clinical presentation? Are they an outlier or atypical?" 
5. Evaluate similarity **Possible methods to try** 
    - Cosine similarity (closer to 1 = similar)
    - Dot product / max(norm(both vectors)) (closer to 1 = similar)
    - Euclidean distance (lower = similar)
    - Mahalanobis distance - accounts for variance and correlation (lower = similar)
6. Use mean-centering to adjust distances to be more interpretable 
    - Distances are more-or-less non-interpretable values anyway, so centering at zero at least provides some context...
6. Do actual clustering algorithm
    - Extract some meaning from the clusters, what factors are they associated with? What features characterize each cluster?
    - See what cluster the new patient would fall into
7. Use UMAP, t-nSE, or PCA to visualize
    - 1D: Mahalanobis distance for how much of an 'outlier' this person is. 
        - KDE plot, with patient value highlighted as vertical line?
    - 2D: Scatter plot with current patient's dot highlighted, and the nearest X number of patients also highlighted
    - Box-plots: of each feature to show where the patient value stacks up among the patients-like-them 
    - Some special visual for the recovery outcomes

