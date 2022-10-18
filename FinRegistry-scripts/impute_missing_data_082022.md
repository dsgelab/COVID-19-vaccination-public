```python
import numpy as np
import csv
from scipy.sparse import csc_matrix
import pandas as pd

datadir = "/data/projects/vaccination_project/data/"
print(datadir)
```


```python
#read in the df and impute missing values by sampling from the same column
#first for training data

inname = '/data/projects/vaccination_project/data/ml_downsampled_data_30082022/vaccination_project_combined_variables_wide_30082022_train_downsampled.csv.newheader'
outname = '/data/projects/vaccination_project/data/ml_downsampled_data_30082022/vaccination_project_combined_variables_wide_30082022_train_downsampled_imputed.csv.newheader'
df = pd.read_csv(inname)
seed = 123
df_imputed = df.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x),replace=True), x))
#save the imputed file
df_imputed.to_csv(outname,index=False)
print("done!")
```


```python
#then imputation for the test set file
inname = datadir+"vaccination_project_combined_variables_wide_30082022_test.csv"
outname = datadir+"vaccination_project_combined_variables_wide_30082022_test_imputed.csv"
df = pd.read_csv(inname)
seed = 123
df_imputed = df.apply(lambda x: np.where(x.isnull(), x.dropna().sample(len(x),replace=True), x))
#save the imputed file
df_imputed.to_csv(outname,index=False)
```

