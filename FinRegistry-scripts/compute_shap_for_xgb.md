```python
import pandas as pd
import random
import csv
#read in the variable names used by the model
all_vars_file = "/data/projects/vaccination_project/data/permute_var_names_082022/predictors_from_each_category_plus_baseline.txt"
with open(all_vars_file,'rt') as infile:
        r = csv.reader(infile,delimiter=',')
        for row in r: all_vars = row
            
train_file = "/data/projects/vaccination_project/data/ml_downsampled_data_30082022/vaccination_project_combined_variables_wide_30082022_train_downsampled.csv.newheader"#"/data/projects/vaccination_project/data/xgboost_train_data/vaccination_project_combined_variables_wide_05072022_train_imputed_downsampled_80-_withheader.csv"
p = 0.05  #keep 5% of the lines
# keep the header, then take only 5% of lines
# if random from [0,1] interval is greater than 0.01 the row will be skipped
df = pd.read_csv(train_file,header=0, skiprows=lambda i: i>0 and random.random() > p,usecols=all_vars)
df
```


```python
#reformat the data for xgboost
import numpy as np
X_train, y_train =  df.drop('COVIDVax',axis=1).values, df.loc[:,'COVIDVax'].values
bg = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
bg.shape
```


```python
#then read on the xgboost model
import pickle
model_file = "/data/projects/vaccination_project/results_30082022_xgboost_noimpute_smallsample/niter_100_ALLVARS_best_xgb_model.pkl"
model = pickle.load(open(model_file,'rb'))
model
```


```python
X_pred = model.predict_proba(X_train)
X_pred
```


```python
import shap
# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.TreeExplainer(model,data=bg,feature_perturbation="interventional",model_output="predict_proba")
shap_values = explainer.shap_values(X_train)
shap_values
```


```python
import matplotlib
from matplotlib import pyplot as plt
#plotting
shap.initjs()
shap.summary_plot(shap_values, X_train,plot_type='bar',feature_names=df.columns[:-1],show=False,max_display=50,plot_size=(8,8))
figdir = "/data/projects/vaccination_project/figures/"
plt.savefig(figdir+'mean_SHAP_per_perdictor_XGBoost_full_model_171022-5-percentage-top50.pdf',dpi=300)
```


```python
shap.dependence_plot('EARNINGS_TOT',shap_values[1],X_train,feature_names=df.columns[:-1],xmax="percentile(99)",xmin="percentile(1)")
```


```python
import statsmodels.api as sm

idx = np.where(df.columns=="EARNINGS_TOT")[0][0]
x = X_train[:,idx]
y_sv = shap_values[1][:,idx]
lowess = sm.nonparametric.lowess(y_sv, x, frac=.3)

_,ax = plt.subplots()
ax.plot(*list(zip(*lowess)), color="red", )

#shap.dependence_plot("Age", shap_values[1], X_train, ax=ax)
shap.dependence_plot('EARNINGS_TOT',shap_values[1],X_train,feature_names=df.columns[:-1],xmax="percentile(99)",
                     ax=ax,interaction_index=None)

```

```python
import numpy as np
#save to a file the mean SHAP values in the 10% sample of training data
outdir = "/data/projects/vaccination_project/data/"

df_pred_shaps = pd.DataFrame()
df_pred_shaps['predictor'] = list(df.columns)[:-1]#header[:-1]
df_pred_shaps['mean |SHAP|'] = np.mean(np.abs(shap_values.values),axis=0)
df_pred_shaps['std |SHAP|'] = np.std(np.abs(shap_values.values),axis=0)
df_pred_shaps['mean SHAP'] = np.mean((shap_values.values),axis=0)

CIs_lower = []
CIs_higher = []

#compute 95% confidence intervals for the mean SHAP values by bootstrapping
n_bootstraps = 1000
rng_seed = 42  # control reproducibility
N = shap_values.values.shape[0]
for j in range(shap_values.shape[1]):
    
    bootstrapped_SHAPs = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, N, N)
        bootstrapped_SHAPs.append(np.mean(np.abs(shap_values.values[indices,:]),axis=0))
    
    sorted_SHAPs = np.array(bootstrapped_SHAPs)
    sorted_SHAPs.sort()

    # Computing the lower and upper bound of the 90% confidence interval
    # You can change the bounds percentiles to 0.025 and 0.975 to get
    # a 95% confidence interval instead.
    confidence_lower = sorted_SHAPs[int(0.025 * len(sorted_SHAPs))]
    confidence_upper = sorted_SHAPs[int(0.975 * len(sorted_SHAPs))]
    
    CIs_lower.append(confidence_lower)
    CIs_higher.append(confidence_upper)
    
df_pred_shaps['mean |SHAP| 95% CI lower'] = CIs_lower
df_pred_shaps['mean |SHAP| 95% CI upper'] = CIs_higher

df_pred_shaps.to_csv(outdir+'mean_SHAP_per_perdictor_XGBoost_full_model_130922-1-percentage.csv',index=False)
df_pred_shaps.sort_values(by='mean |SHAP|')
```

```python
import numpy as np
#save to a file the mean SHAP values in the 10% sample of training data
outdir = "/data/projects/vaccination_project/data/"

df_pred_shaps = pd.DataFrame()
df_pred_shaps['predictor'] = list(df.columns)[:-1]#header[:-1]
df_pred_shaps['mean |SHAP|'] = np.mean(np.abs(shap_values.values),axis=0)
df_pred_shaps['std |SHAP|'] = np.std(np.abs(shap_values.values),axis=0)
df_pred_shaps['mean SHAP'] = np.mean((shap_values.values),axis=0)

CIs_lower = []
CIs_higher = []

#compute 95% confidence intervals for the mean SHAP values by bootstrapping
n_bootstraps = 2000
rng_seed = 42  # control reproducibility
N = shap_values.values.shape[0]
bootstrapped_SHAPs = np.zeros(shape=(shap_values.shape[1],n_bootstraps))

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices = rng.randint(0, N, N)
    bootstrapped_SHAPs[:,i] = np.mean(np.abs(shap_values.values[indices,:]),axis=0)
    
bootstrapped_SHAPs
```

```python
sorted_SHAPs = np.sort(bootstrapped_SHAPs,axis=1)
#sorted_SHAPs.sort(axis=1)
sorted_SHAPs
```

```python
# Computing the lower and upper bound of the 90% confidence interval
# You can change the bounds percentiles to 0.025 and 0.975 to get
# a 95% confidence interval instead.
confidence_lower = sorted_SHAPs[:,int(0.025 * sorted_SHAPs.shape[1])]
confidence_upper = sorted_SHAPs[:,int(0.975 * sorted_SHAPs.shape[1])]
    
#CIs_lower.append(confidence_lower)
#CIs_higher.append(confidence_upper)
    
df_pred_shaps['mean |SHAP| 95% CI lower'] = confidence_lower
df_pred_shaps['mean |SHAP| 95% CI upper'] = confidence_upper

df_pred_shaps.to_csv(outdir+'mean_SHAP_per_perdictor_XGBoost_full_model_100822.csv',index=False)
df_pred_shaps.sort_values(by='mean SHAP')
```

