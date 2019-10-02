#!/usr/bin/env python
# coding: utf-8

# # Gemaps Data Analysis
# 
# 6/27/19
# 
# Diagnostics for Parkinson's
# 
# Warning: drop the `name` column upon feature extraction, it's somewhere in the middle of the csv.  For simplicity I just found it in the excel and deleted it since it was an exact replica of the [0] label column.

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json, sys, regex
from scipy import stats


# # Motor Data
# 
# Original data from patients, taken by Huy. These are the Unified Parkinson Disease Rating Scale (UPDRS III motor score) results to which the feauture extraction data were compared. P value shows insignificant change between groups based off scipy library `ttest` function but hand calculation by Huy seemed to work.

# In[3]:


fig, axes = plt.subplots(figsize=(6,6)) # plot, enlarge


motor = pd.read_csv('Output_Folder/Motor/motor_use.csv')
my_pal = {"DBS off": "tomato", "DBS on": "palegreen"}
sns.boxplot(data=motor, order=["DBS off", "DBS on"], palette=my_pal).set_title('UPDRS Part III Score')

a = motor.iloc[:,0]
b = motor.iloc[:,1]
f = stats.ttest_ind(a,b)
    
f


# Below are data for each of the five patients that were analyzed and assessed for motor scores by Huy. A clear reduction in patient motor school is observed when the DBS is turned on.  Huy preferred the above graph, but I wanted to visualize each patient's data separately for my own sanity.

# In[28]:


fig, ax = plt.subplots()
motor.plot.bar(ax=ax).set_title('UPDRS scores per patient') #stacked=True when Huy made his original data.


# # Gemap Load

# In[5]:


data = pd.read_csv('Output_Folder/Gemaps/Gemaps_features.csv')
names = list(data.columns)
data.head()


# # Split Exp and Ctrl
# 
# Split into DBS_on experimental and DBS_off control.  Allows chisq testing and pipeline implementation.  Also was conducive to our simpler t-test analysis later.

# In[8]:


#split by pd index
off_data = data[::2]
on_data = data[1::2]

#generate json files for the select_features.py file
off_data = off_data.drop(['AudioFile', 'Unnamed: 0'], 1)
on_data = on_data.drop(['AudioFile', 'Unnamed: 0'], 1) 


# # Prep Chi Sq
# 
# This is just straight from the pipeline provided in the voicebook with a few tweaks.  This was not necessary to produce for the data presented at the symposium, but is provided here as a reference for the modified code that works in notebook form.

# In[9]:


X=np.array(on_data)
Y=np.array(off_data)
training=list()
for i in range(len(X)):
    training.append(X[i])
for i in range(len(Y)):
    training.append(Y[i])

# get labels (as binary class outputs)
labels=list()
for i in range(len(X)):
    labels.append(1)
for i in range(len(Y)):
    labels.append(0) 


# # Chi Sq from VoiceBook
# 
# Straight from `select_features.py` from the Voicebook.  Also unnecessary for the symposium data, and is just provided as a commented/tweaked reference guide for future study.

# In[10]:


from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(training, labels, test_size=0.20, random_state=42)
X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train).astype(int)
y_test=np.array(y_test).astype(int)

# normalize features so they are non-negative [0,1], or chi squared test will fail
# it assumes all values are positive 
min_max_scaler = preprocessing.MinMaxScaler()
chi_train = min_max_scaler.fit_transform(X_train)
chi_labels = y_train 

# Select 50 features with highest chi-squared statistics
chi2_selector = SelectKBest(chi2, k=50)
X_kbest = chi2_selector.fit_transform(chi_train, chi_labels)


# # Recursive Feature Elimination, Voicebook
# 
# Recursive feature elmination works by recursively removing 
# attributes and building a model on attributes that remain. 
# It uses model accuracy to identify which attributes
# (and combinations of attributes) contribute the most to predicting the
# target attribute. You can learn more about the RFE class in
# the scikit-learn documentation.  This finds the top 25 features according 
# to the Gemaps algorithm.
# 
# May be desirable to surpress warning for running in Jupyter.
# 
# Recursive feature elmination was supposed to identity features that might work well with the properties of the data set.  However, scanning the p values from the t-test results showed that most of the features were meaningless at the scale of our sample size.

# In[32]:


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

import warnings
warnings.filterwarnings("ignore")

# Top 25 features
model = LogisticRegression() 
rfe = RFE(model, 25)
fit = rfe.fit(X_train, y_train)

# list out number of features and selected features 
print("Num Features: %d"% fit.n_features_) 
print("Selected Features: %s"% fit.support_) 
print("Feature Ranking: %s"% fit.ranking_)


# The following scripts are designed to display all of the features and their associated t stat and p-values.

# In[29]:


fit.support_.size
indx = np.where(fit.support_==True)
indx = np.array(indx)
# indx.flatten()

# apparently a bug in np req np.frombuffer to get 1D array. otherwise the for loop below doesn't work.
indx = np.frombuffer(indx,dtype=int)
indx = indx[::2]


# In[30]:


features = [names[i+1] for i in indx]
#features  #supposedly selected features


# In[33]:


# easier naming for downstream analysis
names = [s.replace('_sma3nz_', ' ') for s in names]
names = [s.replace('stddev', 'std ') for s in names]


# generate t stat results
t_results = dict()
p_vals = list()
t_vals = list()
for i in range(0,88):
    a = off_data.iloc[:,i]
    b = on_data.iloc[:,i]
    f = stats.ttest_ind(a,b,equal_var=True)
    p_vals.append(f.pvalue)
    t_vals.append(f.statistic)
    t_results[names[i+2]] = f
t_results


# Two features were identified as showing a statistically significant relationship between patient data in the DBS off and on state. These features merit further exploration.  One particularly noteworthy finding is that the best features are associated with the mean rising slope of the vocal features in the samples; I would think of this as a first derivative, a measure that does not really depend on the frequency (pitch) regime of the patient's voice. This is great because each patient had a incredibly different tonal quality: some voices were deep while others were high pitched.  The rising slope, however, seems to be more an indication that the voice is suddenly changing in pitch, and thus the measure can be more broadly used across a range of natural vocal frequency ranges.  Again, to test this hypothesis, a more stringent analysis of the frequency dependent (non-derivative features) might be needed, as to assess whether, indeed, those features have high p values.

# In[17]:


df10 = data.iloc[:,11] # t = 2.40, p = .04
df03 = data.iloc[:,4]  # t = 2.08 p = .02

sig_features = pd.concat([df03,  df10], axis = 1, sort=False)
sig_features.head()


# # Plotting only 3, 10
# 
# We combine important features and merge on and off columns into one dataframe to ease seaborn histogram plotting. Data is shown as a histogram with associated p values.

# In[18]:


fig, axes = plt.subplots(figsize=(10,6)) # plot, enlargen

k = 1
for i in [4, 11]:
    
    df = data.iloc[:,i]  # separate on off data
    off = df[::2]
    on = df[1::2]

    # reset index for merge with non NaN values:
    off.reset_index(drop=True, inplace=True)
    on.reset_index(drop=True, inplace=True)

    full = pd.concat([off, on], axis=1)   # concatenate to one 24x2 df
    full.columns.values[[0, 1]] = ['DBS off', 'DBS on']  # rename dataframe columns, for plotting
    
    # Plot 1x2
    plt.subplot(1, 2, k)
    df_title = names[i]
    my_pal = {"DBS off": "tomato", "DBS on": "palegreen"}
    sns.boxplot(data=full, order=["DBS off", "DBS on"], palette=my_pal).set_title(df_title)
    
    # Print Legend
    p_legend = 'p = ' + str('%.3f'%p_vals[i-2]) 
    t_legend = 't-stat = ' + str('%.3f'%t_vals[i-2])
    k += 1
    plt.legend(loc='upper left', labels=[t_legend, p_legend])
    
fig.tight_layout(h_pad=1)
plt.show()


# To put the above graphs into some better context, we show results derived from the first 12 features extracted by the Gemaps library. T stat and p values associated with each are shown in the top left.

# In[19]:


fig, axes = plt.subplots(figsize=(12,20)) # plot, enlargen

for i in range(2,14):
    
    df = data.iloc[:,i]  # separate on off data
    off = df[::2]
    on = df[1::2]
    
    # reset index for merge with non NaN values:
    off.reset_index(drop=True, inplace=True)
    on.reset_index(drop=True, inplace=True)

    full = pd.concat([off, on], axis=1)   # concatenate to one 24x2 df
    full.columns.values[[0, 1]] = ['DBS off', 'DBS on']  # rename dataframe columns, for plotting
    
    plt.subplot(4, 3, i-1)
    df_title = names[i]
    my_pal = {"DBS off": "tomato", "DBS on": "palegreen"}
    sns.boxplot(data=full, order=["DBS off", "DBS on"], palette=my_pal).set_title(df_title)
 
    
    p_legend = 'p = ' + str('%.3f'%p_vals[i-2]) 
    t_legend = 't-stat = ' + str('%.3f'%t_vals[i-2])
    
    plt.legend(loc='upper left', labels=[t_legend, p_legend])
fig.tight_layout(h_pad=1)
plt.show()


# # Wilcoxon Comparison Test
# 
# The Wilcoxon is the non-parametric equivalent to the t-test performed above. It does not assume some normal distribution and thus we felt that its application would be beneficial here because of the small sample size and high frequency regime variability of the individuals who were recorded.  The Wilconxon returns p values that are slightly more promising than the t-test above and are consistent with the results yielded above as well. Should use with caution, because we don't really know if a higher sample size would actually allow some more of the parametric t-test results in for the features to converge anyway.

# In[24]:


# defn dict to pair name with each stat and pval result
wilcoxon_results = dict()
p_vals = list()
w_vals = list()
for i in range(0,88):
    a = off_data.iloc[:,i]
    b = on_data.iloc[:,i]
    f = stats.wilcoxon(a,b)  # comparison
    p_vals.append(f.pvalue)  
    w_vals.append(f.statistic)
    wilcoxon_results[names[i+2]] = f
wilcoxon_results

