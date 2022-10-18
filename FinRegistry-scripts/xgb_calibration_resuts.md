```python
#read in the predictions from XGBoost model
import pandas as pd
from glob import glob
import numpy as np

import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt

from sklearn.calibration import calibration_curve

usecols = ['COVIDVax','xgb_pred_proba']
pred_files =glob('/data/projects/vaccination_project/results_05072022_xgboostcat_imputed_smallsample/niter200_skopt_062022*_test_set_xgb_pred_probas.csv.gz')
print(len(pred_files))
pred_files
```


```python
#plot calibration curves
fig,axs = plt.subplots(4,4,figsize=(15,15))
i = 0
j = 0
for ind in range(len(pred_files)):
    df_test = pd.read_csv(pred_files[ind],usecols=usecols)
    label = pred_files[ind].split('/')[-1].split('.')[0].split('2')[-1][1:-10]
    prob_true, prob_pred = calibration_curve(df_test['COVIDVax'], df_test['xgb_pred_proba'], n_bins=10)
    axs[j,i].plot(np.linspace(0,1),np.linspace(0,1),'--k')
    axs[j,i].plot(prob_pred,prob_true,label=label)
    axs[j,i].legend()
    print(label)
    axs[j,i].set_xlabel('mean predicted value')
    axs[j,i].set_ylabel('fraction of positives')
    axs[j,i].title.set_text(label)
    if i==3:
        i = 0
        j += 1
    else: i+= 1
plt.tight_layout()
```


```python
import pandas as pd
pred_file = "/data/projects/vaccination_project/results_30082022_xgboost_noimpute_smallsample/niter_100_ALLVARS_test_set_xgb_pred_probas.csv.gz"
pred_file = "/data/projects/vaccination_project/results_30082022_xgboost_noimpute_smallsample/_downsampled_testset_ALLVARS_pred_probas.csv.gz"
usecols = ['COVIDVax','xgb_pred_proba']

df_test = pd.read_csv(pred_file,usecols=usecols)
df_test

```python
#plot percentage of vaccinated per centiles of predicted probabilities
import numpy as np

import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt

centiles = [i for i in range(0,101,1)]#[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
scores = np.percentile(df_test['xgb_pred_proba'],centiles)
scores
```

```python
#plot calibration curve
from sklearn.calibration import calibration_curve
figdir = "/data/projects/vaccination_project/figures/"
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)

prob_true, prob_pred = calibration_curve(df_test['COVIDVax'], df_test['xgb_pred_proba'], n_bins=10)
plt.plot(np.linspace(0,1),np.linspace(0,1),'--k')
plt.plot(prob_pred,prob_true)
plt.xlabel('mean predicted value')
plt.ylabel('fraction of positives')
plt.savefig(figdir+'xgb_full_model_downsampled_test_calibration_curve_190922.pdf',dpi=300)
```

```python
#read in the model to get class weights
import pickle
model_file = "/data/projects/vaccination_project/results_30082022_xgboost_noimpute_smallsample/niter_100_ALLVARS_best_xgb_model.pkl"
model = pickle.load(open(model_file,'rb'))
model
```

```python
#recalibrate
prop_class_1 = 0.2

y_pred_odds = 1/((1/df_test['xgb_pred_proba'])-1)
y_adj_odds = y_pred_odds * (prop_class_1 / (1-prop_class_1))
y_adj_probs = 1/ (1+ 1/y_adj_odds)


```


```python
#plot calibration curve
from sklearn.calibration import calibration_curve
figdir = "/data/projects/vaccination_project/figures/"
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}
matplotlib.rc('font', **font)
plt.figure(figsize=(8,8))

prob_true, prob_pred = calibration_curve(df_test['COVIDVax'], df_test['xgb_pred_proba'], n_bins=10)
prob_true_calibrated, prob_pred_calibrated = calibration_curve(df_test['COVIDVax'], y_adj_probs, n_bins=10)
plt.plot(np.linspace(0,1),np.linspace(0,1),'--k')
plt.plot(prob_pred,prob_true,label='noncalibrated')
plt.plot(prob_pred_calibrated,prob_true_calibrated,label='recalibrated')
plt.xlabel('mean predicted value')
plt.ylabel('fraction of positives')
plt.legend()
plt.savefig(figdir+'xgb_full_model_downsampled_test_calibration_curve_recalibrated_190922.pdf',dpi=300)
```

```python
frac_vacced = []
for score in scores:
    frac_vacced.append(np.sum(df_test.loc[df_test['xgb_pred_proba']<=score]['COVIDVax'])/len(df_test.loc[df_test['xgb_pred_proba']<=score]['COVIDVax']))
frac_vacced
```

```python

#plot seaborn bar plots of SHAP values per category
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
plt.figure(figsize=(8,8))

plt.plot(centiles,frac_vacced,linewidth=3)
plt.xlabel('centiles of predicted probabilities')
plt.ylabel('fraction vaccinated')
plt.tight_layout()
plt.savefig(figdir+'xgb_full_model_fraction_vaccinated_per_centiles_of_predicted_probas_130922.pdf',dpi=300)
```

```python
#plot histogram of percentage of vaccinated in each centile bin
mean_vacc_per_bin = []
CI_lower_per_bin = []
CI_upper_per_bin = []
N_sum = 0
Ns = []

n_bootstraps = 2000
rng_seed = 42  # control reproducibility
rng = np.random.RandomState(rng_seed)

for i in range(len(scores)-1):
    #for each centile bin, compute 95% confidence intervals for fraction of non-vaccinated
    #by bootstrapping - i.e. drawing n_bootstraps random samples with replacement
    N = len(df_test.loc[(df_test['xgb_pred_proba']>scores[i]) & (df_test['xgb_pred_proba']<=scores[i+1])]['COVIDVax'])
    N_sum += N
    Ns.append(N)
    mean_vacc_per_bin.append((np.sum(df_test.loc[(df_test['xgb_pred_proba']>=scores[i]) & (df_test['xgb_pred_proba']<=scores[i+1])]['COVIDVax']))/N)
    vacc_stats_bin = df_test.loc[(df_test['xgb_pred_proba']>scores[i]) & (df_test['xgb_pred_proba']<=scores[i+1])]['COVIDVax'].values
    
    means = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement
        indices = rng.randint(0, N, N)
        if len(np.unique(vacc_stats_bin[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        mean = np.mean(vacc_stats_bin[indices])
        means.append(mean)
            
    sorted_means = np.array(means)
    sorted_means.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_means[int(0.025 * len(sorted_means))]
    confidence_upper = sorted_means[int(0.975 * len(sorted_means))]
    CI_lower_per_bin.append(confidence_lower)
    CI_upper_per_bin.append(confidence_upper)
mean_vacc_per_bin = np.array(mean_vacc_per_bin)
confidence_upper = np.array(confidence_upper)
confidence_lower = np.array(confidence_lower)
mean_vacc_per_bin

```

```python
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
plt.figure(figsize=(8,8))
plt.plot(centiles[1:],mean_vacc_per_bin,'o',linewidth=3)
frac_vacc = len(df_test.loc[df_test['COVIDVax']==1])/len(df_test)
print(frac_vacc)
plt.plot(centiles[1:],[frac_vacc for c in centiles[1:]],'--',color='k',linewidth=3)
plt.errorbar(centiles[1:],mean_vacc_per_bin,
             yerr=[mean_vacc_per_bin-CI_lower_per_bin,CI_upper_per_bin-mean_vacc_per_bin],
             fmt='none',color='k',capsize=4)#,elinewidth=4,capsize=6)
plt.xlabel('centile bins of predicted probabilities to not vaccinate')
plt.ylabel('fraction unvaccinated')
plt.ylim([0,1])
plt.tight_layout()
plt.savefig(figdir+'xgb_full_model_fraction_vaccinated_per_centile_bins_of_predicted_probas_130922.pdf',dpi=300)
```

