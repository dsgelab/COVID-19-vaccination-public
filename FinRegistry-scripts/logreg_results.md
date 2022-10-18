```python
from time import time
import pandas as pd
import sys

start = time()

#we want to convert the exactly zero p-values to smallest p-value that R's logistic regression can return
#which is 2.2e-16
min_float = 2.2e-16#sys.float_info.min

#this file contains the variable prevalences overall and among vaccinated
prevalencename = "/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_prevalences.csv" #"/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_062022_prevalences.csv"#"/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_052022_prevalences.csv"
#first read in number of occurrences for each variable
occurrences = {} #key = variable, value = [isBinary,N,N_among_vaxxed,N_NA]
N_omitted = 0
omitted = []
with open(prevalencename,'rt') as infile:
    for row in infile:
        if row.count('columnID')>0: continue
        row = row.strip().split(',')
        isBinary = row[2]
        if isBinary=='True':
            #check that occurrences are not reported if there are less than 6 cases
            #if float(row[3])<6 or float(row[4])<6: occurrences[row[0]] =[isBinary,'NA','NA']
            N = int(row[3])
            N_among_vaxxed = int(row[4])
            #only export data for variables where N or N among vaccinated is at least 6
            if (N_among_vaxxed<6) or (N-N_among_vaxxed<6):
                print("omitting "+row[0]+" | N="+str(N)+", N_among_vaxxed="+str(N_among_vaxxed))
                N_omitted += 1
                omitted.append(row[0])
                continue
            occurrences[row[0]] = [isBinary,int(row[3]),int(row[4]),int(row[5])]
        else: occurrences[row[0]] =[isBinary,'NA','NA',int(row[5])]
            
print('number of omitted variables = '+str(N_omitted))
print('number of kept variables = '+str(len(occurrences)))
print(omitted)
#occurrences['COVIDVax']
```


```python
import csv
#read in the names of endpoints that should be used according to Andrius's work
with open('/data/projects/vaccination_project/data/keep_endpoints_from_Andrius.csv','rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        keep_endpoints = row
        break
print(keep_endpoints[:10])
```


```python
#read in the AUCs
from glob import glob
import pandas as pd
auc_filenames = glob("/data/projects/vaccination_project/results_30082022_lasso_imputed_smallsample_removed_nodata2019/*_auc_CIs.csv")#glob("/data/projects/vaccination_project/results_30082022_lasso_imputed/*_auc_CIs.csv")#glob("/data/projects/vaccination_project/results_31052022_lasso_imputed/*_auc_CIs.csv")

start = time()

results = {} 
#key = variable, value = {}, keys for each i (intercept, SEX, age_july_2021, variable): 'i_oddsratio','i_std_error','i_zval','i_p', also isBinary, AUC, AUC_CI_low, AUC_CI_high
colnames = ['category','N','N_among_vaxxed','N_NA','AUC','AUC_CI_low','AUC_CI_high','isBinary']
variables = []

names = ['intercept','SEX','age_october_2021','variable']
values = ['Coef','minus_95_conf_int','plus_95_conf_int','std_error','p']

for name in names:
    for value in values: colnames.append(name+'_'+value)

for fname in auc_filenames:
    aucs = pd.read_csv(fname,delimiter=',')
    varname = fname.split('/')[-1].split('_auc')[0]
    if varname not in occurrences: continue
    results[varname] = {}
    variables.append(varname)
    results[varname]['AUC'] = aucs['x'].iloc[1]
    results[varname]['AUC_CI_low'] = aucs['x'].iloc[0]
    results[varname]['AUC_CI_high'] = aucs['x'].iloc[2]
    results[varname]['isBinary'] = occurrences[varname][0]
    results[varname]['category'] = varname.split('_')[0]
    results[varname]['N'] = occurrences[varname][1]
    results[varname]['N_among_vaxxed'] = occurrences[varname][2]
    results[varname]['N_NA'] = occurrences[varname][3]
end = time()
print('AUCs read in in '+str(end-start)+" s")
print(len(results))
results['MOTHERTONGUE_Finnish']
```


```python
#then read in the coefficients
coef_filenames = glob("/data/projects/vaccination_project/results_15092022_logreg_removed_nodata2019/*_coefficients.csv")
start = time()

for fname in coef_filenames:
    variable = fname.split('/')[-1].split('_coe')[0]
    if variable not in results: continue
    
    isBinary = True    
    auxrows = []
    #read in the logistic regression model coefficients
    with open(fname,'rt') as coeffile:
        
        for row in coeffile:
            row = row.replace('"','')
            row = row.strip().split(',')
            if row[1].count("Coef")>0: continue
            p = float(row[5])
            if p<min_float: p = min_float#2*10**(-16)
            auxrows.append([row[0].strip('""'),float(row[1]),float(row[2]),float(row[3]),float(row[4]),p])
            if len(auxrows)==4 and isBinary:
                
                for i in range(len(names)):
                    for j in range(len(values)):
                        results[variable][names[i]+'_'+values[j]] = auxrows[i][j+1]
                auxrows = [] 
        if len(auxrows)==3:
            #BIrth registry variables were not adjusted for sex as the data is only for mothers
            addi = 0
            for i in range(len(names)):
                if names[i]=='SEX':
                    for j in range(len(values)):
                        results[variable][names[i]+'_'+values[j]] = 'NA'#auxrows[i][j+1]
                    addi = 1
                else:
                    for j in range(len(values)):
                        results[variable][names[i]+'_'+values[j]] = auxrows[i-addi][j+1]
            auxrows = [] 
                
        
end = time()

print("coefficients read in in "+str(end-start)+" s")
results['MOTHERTONGUE_Finnish']
```


```python
#convert to a dataframe
df = pd.DataFrame.from_dict(results,orient='index',columns=colnames)
df['N'] = pd.to_numeric(df['N'],errors='coerce')
df['N_among_vaxxed'] = pd.to_numeric(df['N_among_vaxxed'],errors='coerce')
df
```


```python
#remove all variables where count among vaccinated is less than or equal to five
df = df.loc[~(df['N_among_vaxxed']<6)]
df
```

```python
#subset the df to include only the variable columns ,and AUC and category, and compute multiple hypothesis testing corrected p-values
from statsmodels.stats.multitest import multipletests
import numpy as np

df_var = df[[i for i in colnames if i.count('variable')>0]+['AUC','AUC_CI_low','AUC_CI_high','category','isBinary','N','N_among_vaxxed','N_NA']]
print(df_var['variable_p'].isna().sum())
Ps_adj = multipletests(df_var.loc[(df_var['isBinary']=='True') & ~(df_var['variable_p'].isna())]['variable_p'],alpha=0.05,method='fdr_bh')[1]
Ps_adj_column = []
P_adj_ind = 0
for i in range(len(df_var['isBinary'])):
    if (df_var['isBinary'].iloc[i]=='True') and not np.isnan(df_var['variable_p'].iloc[i]):
        Ps_adj_column.append(Ps_adj[P_adj_ind])
        P_adj_ind += 1
    else: Ps_adj_column.append(np.nan)
df_var.insert(4,'variable_p_adj',Ps_adj_column,True)#  ['variable_p_adj'] = Ps_adj
df_var = df_var.sort_values(by=['AUC'],ascending=False)
df_var
```

```python
#save the df to a file
outname = "/data/projects/vaccination_project/vaccination_project_variables_coefficients_270922_nodata2019removed.csv"
df_var.to_csv(outname,na_rep='NA',index_label='variable')
df_var.sort_values(by=['AUC','variable_p_adj',],ascending=[False,True])
```

