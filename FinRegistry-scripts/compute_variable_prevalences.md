```python
import numpy as np
import csv
import pandas as pd

inname = "/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_no_Askola.csv.newheader"
#read in the header row
with open(inname,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        header = row
        break
#read in the data in splits of 100 variables
var_splits = []
N = 100
for i in range(0,len(header),N): var_splits.append(header[i:i+N]+['COVIDVax'])
print(var_splits[0])
```


```python
from time import time
#Then compute prevalence, prevalence within vaccinated and number of NAs for each variable
stats_dict = {} #key = column name, value = [isBinary,prevalence, prevalence within vaccinated, number of NAs]
skip_vars = ['FINREGISTRYID','COVIDVax']
N = 10
for i in range(len(var_splits)):
    
    start = time()
    column_slice = pd.read_csv(inname,usecols=var_splits[i],delimiter=',')
    end = time()
    print("split "+str(i)+" read in in "+str(end-start)+" s")
    start = time()
    for var in var_splits[i]:
        if var in skip_vars: continue
        hist = column_slice[var].value_counts()
        if len(hist)<=2:
            #this is a binary variable
            stats_dict[var] = [True,column_slice[column_slice[var]>0][var].count(),column_slice[(column_slice[var]>0) & (column_slice['COVIDVax']<1)][var].count(),column_slice[var].isna().sum()]
            
        else: stats_dict[var] = [False,np.nan,np.nan,column_slice[var].isna().sum()]
        
    end = time()
    print("histo computed in "+str(end-start)+" s")
print(stats_dict)
```


```python
#saving the results into a file
prevalencename = "/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_prevalences.csv"
with open(prevalencename,'wt') as prevalencefile:
    w_prev = csv.writer(prevalencefile,delimiter=',')
    w_prev.writerow(['columnID','index','isBinary','count_1','count_1_among_vaxxed','count_NA'])
    w_prev.writerow(['FINREGISTRYID',0,'no','NA','NA',0])
    for i in range(1,len(header)-1): w_prev.writerow([header[i],i]+stats_dict[header[i]])
```

