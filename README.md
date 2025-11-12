# A Patients-Like-Me approach to aggregating data in clinical management of concussion

**Summary:** This project demonstrates a simple "like-me" approach to matching patients to a reference cohort based on post-concussion demographic and clinical 
characteristics. Using a like-me matched sub-cohort, we can aggregate clinical data for individuals within a reference cohort who are most-similar to the 
presenting patient, providing summary statistics on recovery and other important health information among individuals from the larger cohort who are "as similar as possible" to 
the individual being seen for care.  

## Purpose: 

Complex prediction models, such as neural networks, tree-based classifiers, or simple regressions, can inform patients and clinicians about expected recovery timelines. 
However, these models are limitted by a few factors: 
- Most prediction models are trained on large cohorts with diverse patient and injury characteristics.
    - While usually considered a strength, in some contexts prediction model generalization to a more precise patient population in a real-world setting can be
    limitted - patients are left asking "but do these results apply to *me*?" and clinicians are left wondering if prediction models are accurate in thier patient population.
- Few prediction models can provide clear reasoning for *why* or *how* a prediction is made, or what factors were most influental in a prediction. 
- Prediction models, especially classifiers, can treat medical diseases as a series of binary decisions made in sequence.
    - Actual clinical management of disease is more complex, and clinicians must use all available information to guide decisions to promote long-term health and recovery.  

**Taken together, a quality data science and machine learning approach would support practitioners treating injury, and would supply as much information as possible to guide 
practitioners toward a decision/prediction, rather than making a prediction outright with little context.** 

In this project, I use a "like-me" approach to aggregating data from similar past patients to inform expected recovery outcomes for current patients. In order to do this, we:  
1. Use similarity matching to take the current individuals's clinical presentation (demographics, injury characteristics, etc) and identify past patients from a reference 
cohort with similar characteristics. 
2. Select the N subjects from the reference cohort who are most-similar to the current patient and form a like-me sub-cohort. 
    - N could be defined by the clinician, but by default is flexibly chosen based on how closely the current patient relates to the reference cohort. A larger N is used when 
    patients more closely reflect the reference cohort and a smaller N is used for patients with characteristics that do not closely match the reference cohort. 
3. Aggregate the health and recovery outcomes of subjects in the like-me sub-cohort to inform potential recovery for the current patient.
4. Package this information together in an easy-to-use dashboard so that clinicians and patients can explore how other's "like them" recovered!

## Methods:

1. Generate a reference cohort of patients seen for post-concussion care in our sports medicine clinic (n=558)
    - Limit to just the high level features... this is a proof of concept project, so feasability is most important right now (9 features total)
2. Impute missing data
    - Impution strategy was KNN (K=40) 
    - Works well for numerics, but doesn't have a built-in option for categoricals
        - My desire would be "most frequent" for categorical imputations
        - Without that as a built-in option, KNN already calculates the mean value for the 'K' nearest neighbors
            - Because my categoricals are all binary anyway, I can just round to the nearest integer to get the "most frequent" for my imputed values
3. Normalize the data, Min-Max
    - Need to do this after the imputation, unfortunately
    - Limitation is that variables are not on the same scale during KNN imputation, which can create some bias
4. Apply some scaling to categorical variables
    - When measuring distances (euclidean, for example), the scale of the data is a major factor
        - This is why normalization or standardization is important
    - But, if many binary variables are present, then those variables are all either 0 or 1
        - When numerics are also min-max scaled, then 0 is the lowest value in the dataset and 1 is the largest
        - Therefore, if comparing a numeric and a categorical, a 1-unit difference in the categorical variable is a simple difference of binary category. 
        - But a 1-unit difference in the numeric variable is the entire range of the variable...
        - These things are not equal... theoretically, the 'distance' between a boy and a girl (binary sex categorical feature) is much smaller than that between the minimum 
        and maximum value of any other numeric variable
    - How do we fix this?
        - Scale the categorical variables down after min-max is applied. Multiply the categorical variable by some scalar (I used 0.66), to down-weight the literal distances 
        between the two levels of the feature.
            - In this way, they contribute less (66% as much, actually) to the distance measurements. 
        - Why 0.66? An arbitrary choice on my end. 
            - I could have 'tuned' this value. Or, instead, I could have applied a scalar to each variable (categorical and numeric) based on its univariable 
            association with a recovery outcome, such as time to symptom resolution.
5. Select a single patient at random from the dataset
    - This person will serve as our "example" patient.
    - Remove this person from the cohort and hold their info off to the side - will be used throughout in various ways! 
6. Calculate distance metrics for every person in the cohort (and for our example patient, too)!
    1. Mahalanobis distance from the cohort
        - Using Mahalanobis distance, how far is the subject from the mean of the cohort? 
    2. Mahalanobis distance to the patient
        - Using Mahalanobis distance, how far is the subject from our example patient?
    3. Cosine similarity
        - What is the difference in angle of the subject vector and the example patient vector? Does not account for the magnitude of each vector, though. In a situation where 
        magnitude is important (scale matters and ratios are not particularly important), cosine similarity may not be a good option...
    4. Dot product between vectors, normalized by magnitude
        - Similar to cosine similarity, but is normalized by the magnitude of the largest vector, so magnitude is accounted for. 
    5. Euclidean distance
        - Simple and trust-worthy! 
7. Identify a like-me sub-cohort
    - Select the N subjects from the reference cohort who are closest to the patient based on any of the above distance metrics 
        - I used Mahalanobis distance and euclidean distance to define the closest subjects
        - I used a flexible sub-cohort size (N) based on the patients Mahalanobis distance from the reference cohort
            - A smaller Mahalanobis distance indicates the patient's values are closer to the mean of the reference cohort.
                - For subjects with a smaller Mahalanobis distance to the cohort, a larger sub-cohort size can be used because the patient is very similar to the
                reference cohort. 
            - A larger Mahalanobis distance indicates the patient's values are further from the mean of the refernce cohort.
                - For subjects with a larger Mahalanobis distance to the cohort, a smaller sub-cohort size is necessary because the patient is not as similar to the
                reference cohort and ferwer subjects would be similar to the patient. 
    - N = (10 - Patient Mahalanobis Distance to Cohort)*10 
        - The number of subjects in the like-me sub-cohort is related to the patient's distance to the reference cohort. 
        - Our example patient has a Mahalanobis distance of ~5.5 (around the 75th percentile), so their sub-cohort size would be ~45 subjects. 
8. Plot clinically-relevant information
    - Clinical characteristics for the sub-cohort
    - Recovery outcomes for the sub-cohort 
    - UMAP representations for the entire cohort, with the sub-cohort shown in a different color

## Results and Examples:

### Overview:

We have n=558 in the reference cohort. One subect was randomly selected to serve as the "example patient". In practice, we would know the example patients clinical 
characteristics and would use that to identify a like-me sub-cohort based on similar patients from the reference cohort. We would also use the sub-cohorts recovery 
outcomes to estimate the current patients recovery, based on real data from past persons who are "like them". In this situation, because our example patient comes from the reference cohort, I have the actual recovery outcomes from the example patient and will plot those as well. 

### Distance metrics:

**Figure 1.** I used Mahalabnobis distance, cosine similarity, normalized dot product, and euclidean distance to determine how similar (close) each subject in the reference 
cohort was to the example patient. Those distances are plotted below.

<img src="figs\metrics_plot.png" width=800>

*Would not be shown to the clinician! Mahalanobis distance, euclidean distance, and cosine similarity are positively correlated. The positive relationship between Mahalanobis 
distance and euclidean distance (scatter plot in row 1, column 4), for example, shows that as euclidean distance increases, Mahalanobis distance also increases.*
<br>
<br>

**Figure 2A.** Kernel density plot for each samples' Mahalanobis distance from the mean of the cohort. A lower distance (left side of the x-axis) indicates that the sample is 
closer to the mean of the cohort. Few patients will be very close to the mean in 9-dimensional space. A larger value (right side of the x-axis) indicates that the patient is 
further from the mean of the cohort. The normal distributioin exists because a majoirty of patients are 'about the same' distance from the mean of the cohort. 

<img src="figs\kde_plot_example.png" width=800>

*Clinical interpretation: Most people in the reference cohort are around 4-5 units from the mean of the dataset. That would be 'about' normal, in terms of my patient's 
clinical presentation. A few subjects are <3 units - closer to the left side on the x-axis, so those people are very similar to the mean of the reference cohort. If I have a 
patient toward the middle or to the left of the x-axis here, that would indicate my patient's characteristics are very similar to many of the other patients in the cohort - 
they are quite "typical" in their presentation. My patient, designated by the red dashed line, is slightly to the right on the x-axis. This indicates that my patient has some 
unique aspects to their clinical presentation that leave them further outside the mean of the reference cohort than most other subects in the reference cohort. If they were 
even further to the right, such as having a Mahalanobis distance greater than 6 or 7, then I might wonder if my patient's clinical presentation is too unique - if their 
characteristics are not very similar to the reference cohort and therefore should not be compared to anyone in the cohort using the like-me approach.*  
<br>
<br>

**Figure 2B.** UMAP representation in 2-dimensions of the 9-dimensional clinical characteristics for each subject in the reference cohort.  
**Best UMAP hyperparameters:**

<img src="figs\UMAP_1.png" width=800>

**2nd Best UMAP hyperparameters:**

<img src="figs\UMAP_2.png" width=800>

*Clinical interpretation: This is a 2-dimensional representation of my patient's clinical characteristics. The axes do not really mean anything, they are a summarization of the 
9 other variables we have on my patient. I can see that the like-me sub-cohort is very similar to my patient, because they are all close together in 2-dimensional space.*  
<br>
<br>

**Figure 3A.** Aggregated clinical characteristics for the like-me sub-cohort of samples most similar to the example patient.

<img src="figs\like_me_aggregated_X.png" width=800>

**Figure 3B.** Aggregated categorical clinical characteristics for the like-me sub-cohort of samples most similar to the example patient.

<img src="figs\like_me_aggregated_X_categoricals.png" width=800>

*Clinical interpretation: These are the characteristics that define the like-me sub-cohort.*
<br>
<br>

**Figure 4A.** Aggregated time to symptom resolution for the like-me sub-cohort. The actual patient value is also plotted, though in practice we would not know their time to 
symptom resolution and would instead be using these plots to generate approximate recovery expectations. 

<img src="figs\like_me_aggregated_time_sx.png" width=800>

**Figure 4B.** Aggregated time to return-to-play for the like-me sub-cohort. The actual patient value is also plotted, though in practice we would not know their time to 
return-to-play and would instead be using these plots to generate approximate recovery expectations. 

<img src="figs\like_me_aggregated_time_rtp.png" width=800>

**Figure 4C.** Aggregated proportion of patients with Persisting Symptoms After Concussion (PSaC), or symptoms lasting longer than 28-days, for the like-me sub-cohort.

<img src="figs\like_me_aggregated_PSaC.png" width=800>

*Clinical interpretation: Among ~40 subjects from the like-me sub-cohort, this is their aggregated recovery outcomes. From this, I can set expectations for recovery for my 
current patient, all based on previous patients from my clinic who have similar clinical characteristics.*