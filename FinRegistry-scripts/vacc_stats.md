```python
#read in the preprocessed vaccination project file

inname = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022.csv.gz'
import pandas as pd

df = pd.read_csv(inname,usecols=['FINREGISTRYID','SEX','GEO_MUNICIPALITY_CAT','EDULEVEL_CAT','EDUFIELD_CAT','MOTHERTONGUE_MOTHERTONGUE_CAT','OCCUPATION_CAT',
                                            'first_visit','age_october_2021','COVIDVax'],dtype={'EDULEVEL_CAT':str,'EDUFIELD_CAT':str})
df
```


```python
#get the number of unvaccinated males and females
print('females vaccinated count:')
count_females = len(df.loc[(df['SEX']==1) & (df['COVIDVax']==0)])
print(count_females)
print('females unvaccinated count:')
count_females = len(df.loc[(df['SEX']==1) & (df['COVIDVax']==1)])
print(count_females)
print('females unvaccinated percentage:')
print(100*count_females/len(df.loc[df['SEX']==1]))
print('-----------------------')
print('males vaccinated count:')
count_males = len(df.loc[(df['SEX']==0) & (df['COVIDVax']==0)])
print(count_males)
print('males unvaccinated count:')
count_males = len(df.loc[(df['SEX']==0) & (df['COVIDVax']==1)])
print(count_males)
print('males unvaccinated percentage:')
print(100*count_males/len(df.loc[df['SEX']==0]))
```


```python
#plot number of vaccinated per age and sex
#first convert first_visit column to datetime
df['first_visit'] = pd.to_datetime(df['first_visit'],format='%Y-%m-%d')
#and subset to all vaccinated only
#subset to vaccinated
df_vacc = df.loc[df['COVIDVax']==0]
vacc_counts = df_vacc['GEO_MUNICIPALITY_CAT'].value_counts()
```


```python
#get the count of vaccinated and unvaccinated for each municipality
vax_per_muni = {} #key = name, value = {municipality:,vaxxed:,nonvaxxed:,percentage:}
for m in df['GEO_MUNICIPALITY_CAT'].unique():
    if str(m)=='nan': continue
    vaxxed = len(df.loc[(df['GEO_MUNICIPALITY_CAT']==m) & (df['COVIDVax']==0)])
    non_vaxxed = len(df.loc[(df['GEO_MUNICIPALITY_CAT']==m)])-vaxxed
    print("m="+str(m)+", vaxxed="+str(vaxxed)+",non-vaxxed="+str(non_vaxxed))
    perc = 100*vaxxed/(vaxxed+non_vaxxed)
    vax_per_muni[m] = {'municipality':m,'vaxxed':vaxxed,'non-vaxxed':non_vaxxed,'percentage':perc}
```


```python
vax_per_muni_df = pd.DataFrame.from_dict(vax_per_muni,orient='index')
vax_per_muni_df = vax_per_muni_df.sort_values(by='percentage')
vax_per_muni_df
```


```python
vax_per_muni_df['rank'] = vax_per_muni_df['percentage'].rank()
vax_per_muni_df
```


```python
list(vax_per_muni_df.loc[vax_per_muni_df['percentage']<60]['municipality'])
```


```python
import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
figdir = "/data/projects/vaccination_project/figures/"

plt.figure(figsize=(15,8))

plt.plot(vax_per_muni_df['rank'],vax_per_muni_df['percentage'],'o')
plt.xlabel('Municipalities ranked by vaccination coverage')
plt.ylabel('Vaccination coverage at the end of October 2021')
plt.savefig(figdir+'vacc_coverage_municipalities_ranked_300822.pdf',dpi=300)
```


```python
#get IDs of people living in Askola for excluding them from the study
df_askola = df.loc[df['GEO_MUNICIPALITY_CAT']=='Askola']
df_askola
```


```python
#save these IDs to a file
outfile = '/data/projects/vaccination_project/data/vaccination_project_study_ids_living_in_Askola_082022.csv'
df_askola.to_csv(outfile,sep=',',columns=['FINREGISTRYID'],index=False,header=False)
```


```python
#then remove from df everyone living in Askola
df = df.loc[df['GEO_MUNICIPALITY_CAT']!='Askola']
vacc_counts_per_muni = df_vacc['GEO_MUNICIPALITY_CAT'].value_counts()
vacc_counts_per_muni
```


```python
tot_counts = df['GEO_MUNICIPALITY_CAT'].value_counts()
```

```python
df = df.sort_values(by='first_visit')
df
```


```python
unique_first_visits = df['first_visit'].unique()
```


```python
#get cumulative counts and total counts for different categories
sexes = [0,1]
test_municipalities = ['Helsinki','Turku','Tampere','Oulu','Rovaniemi','Heinola','Lappeenranta','Juuka','Porvoo','Salla']
age_bins = [(30,40),(41,50),(51,60),(61,70),(71,80),(81,90),(91,100)]

xs = []
sex_1_ys = [0]
sex_0_ys = [0]

municipalities_y = {m:[0] for m in test_municipalities}
ages_y = {m:[0] for m in age_bins}

for date in unique_first_visits:
    xs.append(date)
    sex_1_ys.append(sex_1_ys[-1]+len(df.loc[(df['first_visit']==date) & (df['SEX']==1)]))
    sex_0_ys.append(sex_0_ys[-1]+len(df.loc[(df['first_visit']==date) & (df['SEX']==0)]))
    for m in municipalities_y: municipalities_y[m].append(municipalities_y[m][-1]+len(df.loc[(df['first_visit']==date) & (df['GEO_MUNICIPALITY_CAT']==m)]))
    for m in ages_y: ages_y[m].append(ages_y[m][-1]+len(df.loc[(df['first_visit']==date) & (df['age_october_2021']>=m[0]) & (df['age_october_2021']<=m[1])]))
    
```


```python
import numpy as np
plt.figure(figsize=(15,8))
plt.plot(xs,np.array(sex_0_ys[1:])/len(df.loc[df['SEX']==0]),label='male')
plt.plot(xs,np.array(sex_1_ys[1:])/len(df.loc[df['SEX']==1]),label='female')
plt.xticks(rotation=45)
plt.legend()
```

```python
#barplot of vaccinated among males and females
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

print(sex_1_ys[-1]/len(df.loc[df['SEX']==1]))
print(sex_0_ys[-1]/len(df.loc[df['SEX']==0]))

matplotlib.rc('font', **font)
plt.figure(figsize=(8,8))
plt.bar([0,1],[sex_1_ys[-1]/len(df.loc[df['SEX']==1]),sex_0_ys[-1]/len(df.loc[df['SEX']==0])])
plt.ylabel('fraction vaccinated')
plt.xticks(ticks=[0,1],labels=['females','males'])
plt.tight_layout()
plt.savefig(figdir+'vaccinated_males_females_30082022.pdf',dpi=300)
```


```python
#barplot of vaccinated among males and females
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

print(sex_1_ys[-1])
print(sex_0_ys[-1])

matplotlib.rc('font', **font)
plt.figure(figsize=(8,8))
x = np.array([0,1])
width = 0.35

plt.bar(x-width/2,[sex_1_ys[-1],sex_0_ys[-1]],width,label='vaccinated')
plt.bar(x+width/2,[len(df.loc[df['SEX']==1])-sex_1_ys[-1],len(df.loc[df['SEX']==0])-sex_0_ys[-1]],width,label='unvaccinated')
plt.legend()
plt.ylabel('Number of individuals')
plt.ylim([0,1800000])
plt.xticks(ticks=[0,1],labels=['females','males'])
plt.tight_layout()
plt.savefig(figdir+'vaccinated_vs_unvaccinated_males_females_abs_counts_300822.pdf',dpi=300)
```


```python
#cumulative histogram, with percentage on y-axis for municipalities
plt.figure(figsize=(15,8))
for m in municipalities_y:
    plt.plot(xs,np.array(municipalities_y[m][1:])/len(df.loc[df['GEO_MUNICIPALITY_CAT']==m]),label=m,linewidth=4)

plt.ylabel('Vaccination coverage')
plt.xticks(rotation=45)
plt.legend()
plt.savefig(figdir+'cumul_vacc_coverage_per_municipality_examples_300822.pdf',dpi=300)
```


```python
import datetime
#cumulative histogram, with percentage on y-axis for age groups
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)
plt.figure(figsize=(8,8))
for m in ages_y:
    if m==(81,90) or m==(91,100): continue
    plt.plot(xs,np.array(ages_y[m][1:])/len(df.loc[(df['age_october_2021']>=m[0]) & (df['age_october_2021']<=m[1])]),label=str(m[0])+"-"+str(m[1]),
            linewidth=4)

plt.xticks(rotation=45)
plt.ylabel('fraction of age group vaccinated')
plt.legend()
plt.xlim([datetime.date(2021,1,1),datetime.date(2021,10,31)])
plt.savefig(figdir+"cumul_vacc_fraction_per_age_300822_final.pdf",dpi=300)
```

```python
#plot cumulative histogram of all first vaccination doses per sex
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline

figdir = "/data/projects/vaccination_project/figures/"

plt.figure(figsize=(15,8))
ax = sns.ecdfplot(data=df.loc[(df['SEX']==0)],x='first_visit',stat='count',label='male')
ax = sns.ecdfplot(data=df.loc[(df['SEX']==1)],x='first_visit',stat='count',label='female')
plt.legend()
plt.xticks(rotation=45)
plt.savefig(figdir+"cumul_vacc_per_sex_300822_final_proportion.pdf",dpi=300)
```
