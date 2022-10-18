<a id='top'></a>
# Preprocessing for the COVID vaccination project

This notebook assumes that the R scripts in the file with same name than this (but ending with .R), and the Python preprocessing script for infectious diseases and marriage register have been run. Links to different sections of this notebook:

- [Additional minimal phenotype file preprocessing](#minimal_phenotype)
- [Pension registry prerocessing](#pension)
- [Process social HILMO](#social_hilmo)
- [Process social assistance register](#social_assistance)
- [Process birth register](#birth_register)
- [Process socioeconimic status register](#socioeconomic_status)
- [Process occupation register](#occupation_register)
- [Process registry of education](#education)
- [Update the drug purchase file variable names to more interpretable names](#update_drug)
- [Update endpoint variable names to more interpretable names](#update_endpoint)
- [Create variables describing vaccination status of relatives](#relative_vax)
- [Merge all intermediate files into a one wide file used for predictions](#merge_all)
- [Create input variable lists for each logistic regression model](#create_input)

<a id='minimal_phenotype'></a>
## Additional minimal phenotype file preprocessing

We use the filtered minimal phenotype file preprocessed in the R script, but replace mother tongue with the more detailed variable "Äidinkieli" from the DVV relatives registry. For the vaccination project, we will create a separate binary variable for each 152 individual mother tongue found.

Postal codes are preprocessed so that only the first two numbers of the code are kept - there are 3013 individual postal codes in the minimal phenotype file, so this will reduce the number of variables needed. The first two numbers roughly correspond to city/town level division as they define to which sorting center the post is sent.

Date of birth is not kept as we will use the variable age in July 2021 from the Covid vaccination file.

[Go to top of page](#top)


```python
from time import time
import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import csv
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)

#Read in a dictionary for converting the 2 letter language codes to longer names
#This is done to convert the 2-letter codes into more easily understandable format
iso_lang_code_file = "/data/projects/vaccination_project/data/ISO639-language-codes-semicolon.csv"
lang_dict = {'98':'NA','99':'NA','sh':'sh','iw':'iw','mo':'mo'} #key = 2-letter ISO 639-1 code, value = English name of the language
#manually adding some values to the dict that are missing from the ISO definition:
#NA = missing value
#sh = unknown language
#iw = unknown language
#mo = unknown language
with open(iso_lang_code_file,'rt') as infile:
    for row in infile:
        row = row.strip().split(':')
        if row[0].count('Name')>0: continue
        if len(row)==5: lang_dict[row[4]] = row[0].replace(',','_').replace(' ','_')

#Then read in the mother tongues for DVV relatives
dvv_rel_file = "/data/processed_data/dvv/Tulokset_1900-2010_tutkhenk_ja_sukulaiset.txt.finreg_IDsp"

start = time()
mother_tongue = {} #key = ID, value = mother tongue
with open(dvv_rel_file,'rt') as infile:
    for row in infile:
        row = row.strip().split(',')
        if row[0]=='FINREGISTRYID': continue
        ID = row[0]
        if ID not in study_ids: continue
        mt = row[7]
        if ID not in mother_tongue: mother_tongue[ID] = lang_dict[mt]
end = time()
print("Reading in DVV relatives took "+str(end-start)+" s")
print("Number of study population individuals with mother tongue information: "+str(len(list(mother_tongue.keys()))))
```


```python
#add NA to those IDs with missing mother tongue
missing_ids = study_ids - set(mother_tongue.keys())
print("Number of study population IDs missing from DVV relatives: "+str(len(missing_ids)))
for ID in missing_ids: mother_tongue[ID] = 'NA'
```


```python
from operator import itemgetter
#plot histogram of unique values
vals = list(mother_tongue.values())
uniq_vals = list(set(vals))
mother_tongue_vars = []
for u in uniq_vals: mother_tongue_vars.append('mothertongue_'+str(u))
print("Number of unique mother tongues: "+str(len(uniq_vals)))
data = []
for val in uniq_vals: data.append([val,vals.count(val)])
data.sort(key=itemgetter(1),reverse=True)
fig_dims = (24,4)
fig,ax = plt.subplots(figsize=fig_dims)
sns.barplot(x=[x[0] for x in data],y=[x[1] for x in data],ax=ax)
plt.xticks(rotation=90)
plt.yscale('log')
```


```python
#the 20 most frequent languages
for d in range(0,20): print(data[d])
```


```python
#read in all possible postal code values
#notice that we use a more coarse-grained coding that considers only the first two digits
#note also that the leading zeros are missing from the current minimal phenotype file, which needs to be taken into account
mp_file_preproc = "/data/projects/vaccination_project/data/vaccination_project_minimalphenotype_042022.csv"
outfile = "/data/projects/vaccination_project/data/vaccination_project_minimalphenotype_DVV_mt_082022.csv"
mt_column = 4 #mother tongue column index
pc_column = 3 #postal code column index
dob_column = 1 #date of birth column index
rough_postal_codes = {'ZIPCODE_NA':0} #key = first 2 digits of postal code, value = count
mpf_data = []
mpf_data_ids = set()
start = time()
with open(mp_file_preproc,'rt') as infile:
    for row in infile:
        row = row.strip().split(',')
        if row[0]=='FINREGISTRYID': mpf_header = row
        else:
            ID = row[0]
            if ID not in study_ids: continue
            mpf_data_ids.add(ID)
            postal_code = row[pc_column]
            if len(postal_code)<3: postal_code_rough = 'ZIPCODE_NA'
            elif len(postal_code)==3: postal_code_rough = 'ZIPCODE_00'
            elif len(postal_code)==4: postal_code_rough = 'ZIPCODE_0'+postal_code[0]
            else: postal_code_rough = 'ZIPCODE_'+postal_code[:2]
            
            #count the rough postal code occurrences
            if postal_code_rough not in rough_postal_codes: rough_postal_codes[postal_code_rough] = 1
            else: rough_postal_codes[postal_code_rough] += 1
                
            #get the more detailed mother tongue
            if ID in mother_tongue: row[mt_column] = mother_tongue[ID]
            else:
                #just convert the two letter code to longer mother tongue name
                print(row)
                row[mt_column] = lang_dict[row[mt_column]]
                
            #save the data row
            mpf_data.append([row[0],row[2],postal_code_rough]+row[mt_column:])
end = time()
print("Reading in minimal phenotype data took "+str(end-start)+" s")
print("Number of IDs in mpf_data: "+str(len(mpf_data_ids)))
print("Number of study population IDs missing from mpf_data: "+str(len(study_ids-mpf_data_ids)))
```


```python
#read in the municipality corresponding to latest place of residence from Antti's file
import csv
from datetime import datetime

infile = "/data/projects/project_akarvane/geo/living_municipalities.csv"
muni_column = 16
municipality_name_column = 6
ID_column = 1
date_column = 3
latest_municipalities = {} #key = ID, value = [municipality,municipality_name,date]

uniq_municipalities = set(['NA'])
uniq_municipality_names = set(['NA'])
start = time()
with open(infile,'rt') as csvfile:
    r = csv.reader(csvfile,delimiter=',')
    for row in r:
        if len(row[0])<1:
            print(row)
            continue #this is the header row
        #print(row)
        ID = row[ID_column]
        if ID not in study_ids: continue
        date = row[date_column]
        if len(date)<1: continue #if start date is not known, skip
        date = datetime.strptime(date,'%Y-%m-%d')
        municipality = row[muni_column]
        if len(municipality)<2: municipality = 'NA' #empty values are treated as NA
        municipality_name = row[municipality_name_column]

        uniq_municipalities.add('GEO_'+municipality)
        uniq_municipality_names.add('GEO_'+municipality_name)
        
        if ID not in latest_municipalities: latest_municipalities[ID] = [municipality,municipality_name,date]
        elif date>latest_municipalities[ID][2]: latest_municipalities[ID] = [municipality,municipality_name,date]
        
        #if ID not in latest_municipality_names: latest_municipality_names[ID] = [municipality_name,date]
        #elif date>latest_municipality_names[ID][1]: latest_municipality_names[ID] = [municipality_name,date]

end = time()
print("Latest municipalities read in in "+str(end-start)+" s")
print('Number of study population IDs with place of residence entry: '+str(len(list(latest_municipalities.keys()))))
missing_ids = study_ids-set(latest_municipalities.keys())
print('Number of study population IDs with no place of residence entry: '+str(len(missing_ids)))
```


```python
#plot histogram of unique municipalities
vals = [m[0] for m in latest_municipalities.values()]
uniq_vals = set(vals)
municipality_vars = uniq_vals

print("Number of unique municipalities: "+str(len(uniq_vals)))
data = []
for val in uniq_vals: data.append([val,vals.count(val)])
data.sort(key=itemgetter(1),reverse=True)
print(data[:30])
fig_dims = (24,4)
fig,ax = plt.subplots(figsize=fig_dims)
sns.barplot(x=[x[0] for x in data],y=[x[1] for x in data],ax=ax)
plt.xticks(rotation=90)
plt.yscale('log')
```


```python
#plot histogram of unique municipality names
vals = [m[1] for m in latest_municipalities.values()]
uniq_vals = set(vals)
municipality_vars = uniq_vals
#for u in uniq_vals: municipality_vars.append('mothertongue_'+str(u))
print("Number of unique municipality names: "+str(len(uniq_vals)))
data = []
for val in uniq_vals: data.append([val,vals.count(val)])
data.sort(key=itemgetter(1),reverse=True)
print(data[:30])

fig_dims = (24,4)
fig,ax = plt.subplots(figsize=fig_dims)
sns.barplot(x=[x[0] for x in data],y=[x[1] for x in data],ax=ax)
plt.xticks(rotation=90)
plt.yscale('log')
```


```python
#add NA to those IDs with missing place of residence
for ID in missing_ids: latest_municipalities[ID] = ['NA','NA','NA']
```


```python
#plot histogram of unique municipalities
vals = [m[0] for m in latest_municipalities.values()]
uniq_vals = set(vals)
municipality_vars = uniq_vals
print("Number of unique municipalities: "+str(len(uniq_vals)))
data = []
for val in uniq_vals: data.append([val,vals.count(val)])
data.sort(key=itemgetter(1),reverse=True)
print(data[:30])
fig_dims = (24,4)
fig,ax = plt.subplots(figsize=fig_dims)
sns.barplot(x=[x[0] for x in data],y=[x[1] for x in data],ax=ax)
plt.xticks(rotation=90)
plt.yscale('log')
```


```python
#create indicator variables for each municipality
vals = list(mother_tongue.values())
uniq_vals = list(set(vals))
mpf_header_new = ['FINREGISTRYID','SEX','MOTHERTONGUE_MOTHERTONGUE_CAT','GEO_MUNICIPALITY_CAT','MOTHERTONGUE_NOFINSWE','GEO_MUNICIPALITY_SMALLTOWN']+[m for m in uniq_municipalities]+['MOTHERTONGUE_'+v for v in uniq_vals]+['SES_'+m for m in mpf_header[mt_column+1:-1]]+['DRUG_'+mpf_header[-1]]
#print(mpf_header_new)
```


```python
#rename NA -> GEO_NA
NA_ind = mpf_header_new.index('NA')
mpf_header_new[NA_ind] = 'GEO_NA'
```


```python
#next save the minimal phenotype file with the new postal code and mother tongue features

large_cities = ['Helsinki','Espoo','Tampere','Vantaa','Oulu','Turku','Jyväskylä','Kuopio','Lahti'] #list of cities with >100k inhabintans in Finland

ever_index = mpf_header_new.index('SES_ever_married')
mt_cat_index = mpf_header_new.index('MOTHERTONGUE_MOTHERTONGUE_CAT')
mt_nofinswe_index = mpf_header_new.index('MOTHERTONGUE_NOFINSWE')
geo_cat_index = mpf_header_new.index('GEO_MUNICIPALITY_CAT')
geo_smalltown_index = mpf_header_new.index('GEO_MUNICIPALITY_SMALLTOWN')

all_nonNA_mt_inds = [mpf_header_new.index(name) for name in mpf_header_new if (name[-4:]!='_CAT' and name.count('MOTHERTONGUE_')>0 and name.count('MOTHERTONGUE_NA')==0)]
all_nonNA_geo_inds = [mpf_header_new.index(name) for name in mpf_header_new if (name[-4:]!='_CAT' and name.count('GEO_')>0 and name.count('GEO_NA')==0)]

start = time()
with open(outfile,'wt') as of:
    w = csv.writer(of,delimiter=',')
    w.writerow(mpf_header_new)
    for row in mpf_data:
        new_row = ['0' for i in range(len(mpf_header_new))]
        new_row[:2] = row[:2] #ID, sex
        ID = row[0]
        if ID not in latest_municipalities: municipality = 'NA'
        else: municipality = latest_municipalities[ID][0]
        #new_row[postal_cat_index] = row[2].split('_')[1] #postal code as a categorical variable
        new_row[mt_cat_index] = row[3] #mother tongue as a categorical variable
        #Then Finnish and Swedish speakers vs the rest:
        if row[3]=='NA': new_row[mt_nofinswe_index] = 'NA'
        elif row[3]!='Finnish' and row[3]!='Swedish': new_row[mt_nofinswe_index] = 1
        else: new_row[mt_nofinswe_index] = 0
        new_row[geo_cat_index] = municipality #latest municipality as categorical variable
        #Living in a <100k inhabintant town vs >100k inhabitant town
        if municipality=='NA': new_row[geo_smalltown_index] = 'NA'
        elif municipality in large_cities: new_row[geo_smalltown_index] = 0
        else: new_row[geo_smalltown_index] = 1
        #post_index = mpf_header_new.index(row[2])
        #new_row[post_index] = 1
        mt_index = mpf_header_new.index('MOTHERTONGUE_'+row[3])
        new_row[mt_index] = 1
        #If value is missing, setting all mothe tongue variables to NA
        if row[3]=='NA':
            for ind_NA in all_nonNA_mt_inds: new_row[ind_NA] = 'NA'
        #print('municipality: '+municipality)
        geo_index = mpf_header_new.index('GEO_'+municipality)
        new_row[geo_index] = 1
        #If value is missing, setting all geo variables to NA
        if municipality=='NA':
            for ind_NA in all_nonNA_geo_inds: new_row[ind_NA] = 'NA'
        new_row[ever_index:] = row[4:]
        w.writerow(new_row)
end = time()
print("Minimal phenotype updated in "+str(end-start)+" s")
```

<a id='pension'></a>
## Pension registry preprocessing

From the pension registry, only the monthly income is used. If no income is found, it is marked as missing (NA). Zero income is treated as zero, not as missing.

Income is divided into percentiles based on the income distribution. Everyone with exactly 0 income is excluded, from computing the deciles. The columns of the intermediate file are:

- 1. FINREGISTRYID
- 2. EARNINGS_TOT - Total earnings in 2019
- 3. EARGNINGS_CAT - Categorical variable of the earnings deciles (1-10) in 2019
- 4. EARNINGS_1DEC - Indicator variable of belonging to the first earning decile
- 5. EARNINGS_2DEC - Indicator variable of belonging to the second earning decile
- 6. EARNINGS_3DEC - Indicator variable of belonging to the third earning decile
- 7. EARNINGS_4DEC - Indicator variable of belonging to the fourth earning decile
- 8. EARNINGS_5DEC - Indicator variable of belonging to the fifth earning decile
- 9. EARNINGS_6DEC - Indicator variable of belonging to the sixth earning decile
- 10. EARNINGS_7DEC - Indicator variable of belonging to the seventh earning decile
- 11. EARNINGS_8DEC - Indicator variable of belonging to the eight earning decile
- 12. EARNINGS_9DEC - Indicator variable of belonging to the ninth earning decile
- 13. EARNINGS_10DEC - Indicator variable of belonging to the tenth earning decile
- 14. EARNINGS_NA - Indicator variable indicating missing data

[Go to top of page](#top)


```python
earnings_name = "/data/processed_data/etk_pension/vuansiot_2022-05-12.feather"
#read in the indexed earnings
import pandas as pd
df_earnings = pd.read_feather(earnings_name,columns=['vuosi','id','vuosiansio_indexed'])
#subset to select only year 2019 as it is the last "normal" year
df_earnings = df_earnings.loc[df_earnings['vuosi']==2019]
df_earnings
```


```python
#compute the sum of earnings for year 2019 for each ID
tot_earnings_in_2019 = df_earnings.groupby(by='id')['vuosiansio_indexed'].sum()
tot_earnings_in_2019
```

```python
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)
#get the ids for which we do not have income information
missing_ids = study_ids - set(list(tot_earnings_in_2019.keys()))
extra_ids = set(list(tot_earnings_in_2019.keys())) - study_ids
print("Number of study population IDs with missing income data: "+str(len(missing_ids)))
print("Number of IDs with in income data that are not in the study population: "+str(len(extra_ids)))
```


```python
from time import time
start = time()
tot_earnings_in_2019_study_pop = {} #key=ID, value=total earnings
for ID,value in tot_earnings_in_2019.items():
    if ID in study_ids: tot_earnings_in_2019_study_pop[ID] = value
end = time()
print('Removing the extra IDs took '+str(end-start)+" s")
print('Number of study population IDs with income: '+str(len(list(tot_earnings_in_2019_study_pop.keys()))))
```


```python
tot_earnings_in_2019_study_pop_df = pd.DataFrame.from_dict(tot_earnings_in_2019_study_pop,orient='index')
tot_earnings_in_2019_study_pop_df.sort_values(by=0)
```


```python
#exclude everyone with 0 income
tot_earnings_in_2019_nonzero = tot_earnings_in_2019_study_pop_df.loc[tot_earnings_in_2019_study_pop_df[0]>0]
print(str(len(tot_earnings_in_2019_study_pop_df)-len(tot_earnings_in_2019_nonzero))+" individuals with 0 income for 2019")
tot_earnings_in_2019_nonzero.sort_values(by=0)
```


```python
#compute quantiles and use these as a categorical variable
#old quantiles without removing 0s were:
#quantile 0.1 : 4433.068625
#quantile 0.2 : 11147.941
#quantile 0.3 : 19362.38325
#quantile 0.4 : 26843.89925
#quantile 0.5 : 31846.662874999998
#quantile 0.6 : 36435.13175
#quantile 0.7 : 41706.778499999986
#quantile 0.8 : 49025.52449999999
#quantile 0.9 : 61978.06

quantiles = {} #key = quantile, value = minimum value for belonging to the quantile
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    q = tot_earnings_in_2019_nonzero.quantile(i)
    print("quantile "+str(i)+" : "+str(q))
    quantiles[str(i)] = q
```


```python
#plot the histogram of total earnings in 2019 for people that received any salary
from time import time
import matplotlib
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns

bins = [0]
for q in ["0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9"]: bins.append(quantiles[q].values[0])
bins.append(999999999)
print(bins)
sns.histplot(x=tot_earnings_in_2019_nonzero[0])
```

```python
from time import time
start = time()

outname = "/data/projects/vaccination_project/data/vaccination_project_yearly_earnings_082022.csv"
#for each ID, define in which salary decile they are
header = ['FINREGISTRYID','EARNINGS_TOT','EARNINGS_CAT','EARNINGS_1DEC','EARNINGS_2DEC','EARNINGS_3DEC','EARNINGS_4DEC',
          'EARNINGS_5DEC','EARNINGS_6DEC','EARNINGS_7DEC','EARNINGS_8DEC','EARNINGS_9DEC','EARNINGS_10DEC','EARNINGS_NA']
deciles = []
with open(outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(header)
    for index,row in tot_earnings_in_2019_study_pop.items():
        ID = index#row[0]
        earning = row
        quantile = 1
        for i in range(len(bins)):
            
            q = bins[i+1]
            if earning>q: quantile += 1
            else: break
        new_row = [0 for i in range(len(header))]
        new_row[0] = ID
        new_row[1] = earning
        new_row[2] = quantile
        new_row[header.index('EARNINGS_'+str(quantile)+"DEC")] = 1
        w.writerow(new_row)
    #then write entries for missing IDs
    for ID in missing_ids:
        w.writerow([ID,'NA','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA','NA',1])
end = time()
print('Earnings file written in '+str(end-start)+" s")
```

<a id='social_hilmo'></a>
## Process social HILMO

Columns to include in the final wide file:
- 1. FINREGISTRYID (TNRO)
- 2.-21. SOCIALWELFARE_TYPEOFCARE_i : PALA (20 unique values, including NA, each coded as its own columns 1/0)
- 22.-50. SOCIALWELFARE_MAINTREATMENTREASON_i : TUSYY1 (29 unique values, including NA, each coded as its own columns 1/0)
- 51. SOCIALWELFARE_LONGTERMCARE : PITK
- 52. SOCIALWELFARE_TREATMENTDAYSUM : HOITOPV (sum of total treatment days over all entries for an ID)
- 53. SOCIALWELFARE_TREATMENTDAYSUM_NA : indicator for individuals missing from social HILMO

NOTE! TUSYY1 codes 0, 13 and 30 are discarded due to low occurrence in the dataset (0: 19 occurrences, 13: 1 occurrence, 30: 1 occurrence). Also TUSYY1 codes are treated as integers as according to HILMO documentation integer and decimal representations of the same number should correspond to the same code.

[Go to top of page](#top)


```python
from os import system
from time import time
import csv
import sys

start = time()

#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)

#first create an intermediate file with only the columns of interest
shil_file = '/data/processed_data/thl_soshilmo/thl2019_1776_soshilmo.csv.finreg_IDsp'
tmpfile = '/home/thartone/vaccination_project/tmp_data/thl2019_1776_soshilmo.csv.finreg_IDsp_vacc_tmp.csv'
system("cut -d ',' -f1,7,15,28,42,39,11 "+shil_file+" > "+tmpfile)
#the tmp file contains the following columns:
#TNRO,PALA,TUPVA,TUSYY1,PITK,IKAT,HOITOPV
end = time()
print("tmp file created in "+str(end-start)+" s")

start = time()
#read in the tmpfile
shil = {} #key = ID, value = list of following entries: [set of unique PALA,set of unique TUSYY1,max of PITK,sum of HOITOPV]
Nkeys = 0
Nrow = 0
uniq_TUSYY1 = set() #all unique TUSYY1 codes used as variables
uniq_PALA = set() #all unique PALA codes used as variables

with open(tmpfile,'rt') as infile:
    for row in infile:
        Nrow += 1
        row = row.split(',')

        if row[0]=='TNRO': continue
        ID = row[0]
        if ID not in study_ids: continue
        PALA = row[1]
        if len(PALA)<1: PALA = 'NA'
        if PALA!='NA': PALA = int(float(PALA))
        TUSYY1 = row[3].strip(',').strip('"')
        if len(TUSYY1)<1: TUSYY1 = 'NA' #if value is missing, mark it as NA
        elif TUSYY1!='NA':
            TUSYY1 = int(float(TUSYY1))
            if TUSYY1 in [0,13,30]: TUSYY1 = 'NA' #remove codes 0, 13 and 30 because they are so rare
        PITK = row[4].strip('\n')
        if len(PITK)<1: PITK = 0 #if value is missing mark as 0
        elif PITK=='E': PITK = 0
        elif PITK=='K': PITK = 1
        else: PITK = 'NA' #there is in total approx. 100 entries where PITK is a decimal number,
        #these are regarded as errors
        
        HOITOPV = row[6].strip('\n')
        if len(HOITOPV)>0: HOITOPV = float(HOITOPV)
        else: HOITOPV = 0.0
        
        uniq_PALA.add(str(PALA))
        uniq_TUSYY1.add(str(TUSYY1))
        
        if ID not in shil:
            if PITK!='NA': shil[ID] = [set([PALA]),set([TUSYY1]),PITK,HOITOPV]
            else: shil[ID] = [set([PALA]),set([TUSYY1]),0,HOITOPV]
            Nkeys += 1
        else:
            shil[ID][0].add(PALA)
            shil[ID][1].add(TUSYY1)
            if PITK!='NA':
                shil[ID][2] = max([shil[ID][2],PITK])
            shil[ID][3] += HOITOPV

end = time()
print("Data read in in "+str(end-start)+" s")
print("unique PALA codes: "+str(uniq_PALA))
print("unique TUSYY1 codes: "+str(uniq_TUSYY1))
```


```python
#get the ids for which we do not have social hilmo data
missing_ids = study_ids - set(list(shil.keys()))
print("Number of study population IDs with missing social hilmo data: "+str(len(missing_ids)))

```


```python
start = time()
outname = '/data/projects/vaccination_project/data/vaccination_project_soshilmo_wide_082022.csv'
#create header for the output file
header = ['FINREGISTRYID']+['SOCIALWELFARE_TYPEOFCARE_'+str(i) for i in sorted(list(uniq_PALA))]+['SOCIALWELFARE_MAINTREATMENTREASON_'+str(i) for i in sorted(list(uniq_TUSYY1))]+['SOCIALWELFARE_LONGTERMCARE','SOCIALWELFARE_TREATMENTDAYSUM','SOCIALWELFARE_TREATMENTDAYSUM_NA']
nonNA_typeofcare_inds = [header.index(name) for name in header if (name.count('TYPEOFCARE')>0)]
nonNA_maintreatmentreason_inds = [header.index(name) for name in header if (name.count('MAINTREATMENTREASON')>0)]

with open(outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(header)
    for ID in shil:
        row = [0 for i in range(len(header))]
        row[0] = ID
        for PALA in shil[ID][0]:
            row[header.index('SOCIALWELFARE_TYPEOFCARE_'+str(PALA))] = 1
            if PALA=='NA':
                for NA_ind in nonNA_typeofcare_inds: row[NA_ind] = 'NA'
        for TUSYY1 in shil[ID][1]:
            row[header.index('SOCIALWELFARE_MAINTREATMENTREASON_'+str(TUSYY1))] = 1
            if TUSYY1=='NA':
                for NA_ind in nonNA_maintreatmentreason_inds: row[NA_ind] = 'NA'
        row[-3] = shil[ID][2]
        row[-2] = shil[ID][3]
        w.writerow(row)
    #then write entries for the missing IDs
    for ID in missing_ids: 
        row = [ID]+['NA' for i in range(len(header)-1)]
        row[-1] = 1
        w.writerow(row)
end = time()
print("Output written in "+str(end-start)+" s")
```

<a id='social_assistance'></a>
## Process social assistance register

Columns to include in the final wide file:
- 1. FINREGISTRYID (TNRO)
- 2. SOCIALASSISTANCE_SUPPORTMONTHS : TUKIKUUKAUSIA (use sum of all longitudinal entries per ID)
- 3. SOCIALASSISTANCE_INCOMESUPPORTEUR : VARS_TOIMEENTULOTUKI_EUR (use sum of all longitudinal entries per ID)
- 4. SOCIALASSISTANCE_INCOMESUPPORTMONTHS : VARS_TOIMEENTULOTUKI_KK (use sum of all longitudinal entries per ID)
- 5. SOCIALASSISTANCE_NA : Person is missing from social assistance register

[Go to top of page](#top)


```python
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)
```


```python
start = time()
#first create an intermediate file with only the columns of interest
sar_file = '/data/processed_data/thl_social_assistance/3214_FinRegistry_toitu_MattssonHannele07122020.csv.finreg_IDsp'
tmpfile = '/data/projects/vaccination_project/data/3214_FinRegistry_toitu_MattssonHannele07122020.csv.finreg_IDsp_vacc_tmp.csv'
system("cut -d ';' -f1,2,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21 "+sar_file+" > "+tmpfile)
#tmp file contains columns: TNRO;TILASTOVUOSI;TAMMI;HELMI;MAALIS;HUHTI;TOUKO;KESA;HEINA;ELO;SYYS;LOKA;MARRAS;JOULU;TUKIKUUKAUSIA;VARS_TOIMEENTULOTUKI_EUR;VARS_TOIMEENTULOTUKI_KK
end = time()
print("tmp file created in "+str(end-start)+" s")

start = time()
#read in the tmpfile
missing_IDs = 0
sar = {} #key = ID, value = list of following entries: [sum of TUKIKUUKAUSIA,sum of VARS_TOIMEENTULOTUKI_EUR,sum of VARS_TOIMEENTULOTUKI_KK]

with open(tmpfile,'rt') as infile:
    for row in infile:
            
        row = row.split(';')
        if row[0]=='TNRO': continue
        ID = row[0]
        if ID not in study_ids: continue
                
        TUKIKUUKAUSIA = int(row[14].strip())
           
        VARS_TOIMEENTULOTUKI_EUR = row[15].strip()
        if len(VARS_TOIMEENTULOTUKI_EUR)<1:
            VARS_TOIMEENTULOTUKI_EUR = 0 #if value is missing, treat as 0
        else:
            VARS_TOIMEENTULOTUKI_EUR = float(VARS_TOIMEENTULOTUKI_EUR)
        VARS_TOIMEENTULOTUKI_KK = row[16].strip()
        if len(VARS_TOIMEENTULOTUKI_KK)<1: 
            VARS_TOIMEENTULOTUKI_KK = 0 #if value is missing, treat as 0
        else: 
            VARS_TOIMEENTULOTUKI_KK = int(VARS_TOIMEENTULOTUKI_KK)
        if ID not in sar: sar[ID] = [TUKIKUUKAUSIA,VARS_TOIMEENTULOTUKI_EUR,VARS_TOIMEENTULOTUKI_KK]
        else:
            sar[ID][0] += TUKIKUUKAUSIA
            sar[ID][1] += VARS_TOIMEENTULOTUKI_EUR
            sar[ID][2] += VARS_TOIMEENTULOTUKI_KK

end = time()
missing_ids = study_ids - set(list(sar.keys()))
print("Number of study population IDs missing from social assistance register "+str(len(missing_ids))+".")
print("Data read in in "+str(end-start)+" s")
```


```python
start = time()
outname = '/data/projects/vaccination_project/data/vaccination_project_socialassistanceregister_wide_082022.csv'
#create header for the output file
header = ['FINREGISTRYID','SOCIALASSISTANCE_SUPPORTMONTHS','SOCIALASSISTANCE_INCOMESUPPORTEUR','SOCIALASSISTANCE_INCOMESUPPORTMONTHS','SOCIALASSISTANCE_NA']
with open(outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(header)
    for ID in sar: w.writerow([ID]+sar[ID]+[0])
    #then write entries for the missing IDs
    for ID in missing_ids: w.writerow([ID,'NA','NA','NA',1])
end = time()
print("Output written in "+str(end-start)+" s")
```

<a id='birth_register'></a>
## Process birth register

Columns to include in the final file:
- 1. FINREGISTRYID (AITI_TNRO)
- 2. BIRTH_COHABITING : AVOLIITTO, 3 unique values, yes=1 (1 in original file), no=0 (2 in original file), NA=NA (9 in original file)
- 3. BIRTH_MISCARRIAGES : KESKENMENOJA, count of miscarriages
- 4. BIRTH_TERMINATEDPREGNANCIES : KESKEYTYKSIA, count of terminated pregnancies
- 5. BIRTH_TERMINATED_PREGNANCIES : ULKOPUOLISIA, count of ectopic pregnancies
- 6. BIRTH_STILLBORNS : KUOLLEENASYNT, count of previous births with at least one stillborn infant
- 7. BIRTH_SMOKE_NO, binary (no smoking during pregnancy=1, other=0),TUPAKOINTITUNNUS value 1 in original data
- 8. BIRTH_SMOKE_QUIT, binary (quit smoking during 1. trimester=1, other=0), TUPAKOINTITUNNUS value 2 in original data
- 9. BIRTH_SMOKE_YES, binary (smoked after 1. trimester=1, other=0), TUPAKOINTITUNNUS values 3-4. NOTE! available only starting from 2017
- 10. BIRTH_SMOKE_NA, binary (No info on smoking=1, other=0), TUPAKOINTITUNNUS value 9 in original data.
- 11. BIRTH_IVF, binary (yes=1,no=0)
- 12. BIRTH_TROMBOSISPROFYLAXY : TROMBOOSIPROF, binary (yes=1,no=0)
- 13. BIRTH_ANEMIA, binary (yes=1,no=0)
- 14. BIRTH_GLUCOSETESTED : SOKERI_PATOL, binary (yes=1,no=0)
- 15. BIRTH_NOANALGESIA : EI_LIEVITYSTA, binary (yes=1,no=0)
- 16. BIRTH_ANALGESIA_NA : EI_LIEVITYS_TIETOA, binary (yes=1,no=0)
- 17. BIRTH_ARTIFICIALLYINITIATED : KAYNNISTYS, binary (yes=1,no=0)
- 18. BIRTH_PROMOTED : EDISTAMINEN, binary (yes=1,no=0)
- 19. BIRTH_PUNCTURE : PUHKAISU, binary (yes=1,no=0)
- 20. BIRTH_OXYTOCIN : OKSITOSIINI, 3 unique values, yes=1 (1 in original file), no=0 (0 in original file), NA=NA (9 in original file)
- 21. BIRTH_PROSTAGLANDIN : PROSTAGLANDIINI, binary (yes=1,no=0)
- 22. BIRTH_MANUALEXTRACTION : ISTUKANIRROITUS, binary (yes=1,no=0)
- 23. BIRTH_UTERINESCRAPING : KAAVINTA, binary (yes=1,no=0)
- 24. BIRTH_SUTURING : OMPELU, binary (yes=1,no=0)
- 25. BIRTH_GBSMED : GBS_PROFYLAKSIA, binary (yes=1,no=0)
- 26. BIRTH_MOTHERANTIBIOTICS : AIDIN_ANTIBIOOTTIHOITO, binary (yes=1,no=0)
- 27. BIRTH_BLOODTRANSFUSION : VERENSIIRTO, binary (yes=1,no=0)
- 28. BIRTH_CIRCUMCISION : YMPARILEIKKAUKSEN_AVAUS, binary (yes=1,no=0)
- 29. BIRTH_HYSTERECTOMY : KOHDUNPOISTO, binary (yes=1,no=0)
- 30. BIRTH_EMBOLISATION : EMBOLISAATIO, binary (yes=1,no=0)
- 31. BIRTH_VAGINAL, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS value 1 in original file
- 32. BIRTH_BREECH, binary (yes=1, no=0), SYNNYTYSTPATUNNUS value 2 in original file
- 33. BIRTH_FORCEPS, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS value 3 in original file
- 34. BIRTH_VACUUM, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS value 4 in original file
- 35. BIRTH_PLANNEDC, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS value 5 in original file
- 36. BIRTH_URGENTC, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS, value 6 in original file
- 37. BIRTH_EMERGENCYC, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS, value 7 in original file
- 38. BIRTH_OTHERC, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS, value 8 in original file
- 39. BIRTH_NA, binary (yes=1, no=0), SYNNYTYSTAPATUNNUS 9 in original file
- 30. BIRTH_PLACENTAPRAEVIA : ETINEN, , binary (yes=1,no=0)
- 31. BIRTH_ABLATIOPLACENTAE : ISTIRTO, binary (yes=1,no=0)
- 32. BIRTH_ECLAMPSIA : RKOURIS, binary (yes=1,no=0)
- 33. BIRTH_SHOULDERDYSTOCIA : HARTIADYSTOKIA, binary (yes=1,no=0)
- 34. BIRTH_ASPHYXIA : ASFYKSIA, binary (yes=1,no=0)
- 35. BIRTH_BREECH : PERATILA, binary (yes=1,no=0)
- 36. BIRTH_S_LIVE, binary (yes=1, no=0), SYNTYMATILATUNNUS value 1 in original file
- 37. BIRTH_S_DIEDBEFORE, binary (yes=1, no=0), SYNTYMATILATUNNUS value 2 in original file
- 38. BIRTH_S_DIEDDURING, binary (yes=1, no=0), SYNTYMATILATUNNUS value 3 in original file
- 39. BIRTH_S_DIEDUNKNOWN, binary (yes=1, no=0), SYNTYMATILATUNNUS value 4 in original file
- 40. BIRTH_S_NA, binary (yes=1, no=0), SYNTYMATILATUNNUS value 9 in original file
- 41. BIRTH_REG_NA, binary (yes=1, no=0), indicates whether the individual is missing from birth registry


```python
from time import time
from os import system
#columns to use 1,6,10,14,15,16,18,23,40,63,64,66,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,97,99,100,101,102,103,104,117,175
start = time()
#first create an intermediate file with only the columns of interest
birth_file = '/data/processed_data/thl_birth/THL2019_1776_synre.csv.finreg_IDsp'
tmpfile = '/data/projects/vaccination_project/data/THL2019_1776_synre.csv.finreg_IDsp_vacc_tmp.csv'
system("cut -d ',' -f1,6,10,14,15,16,18,23,40,63,64,66,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,97,99,100,101,102,103,104,117,175 "+birth_file+" > "+tmpfile)
#columns of tmp file: AITI_TNRO,AITI_IKA,AVOLIITTO,KESKENMENOJA,KESKEYTYKSIA,ULKOPUOLISIA,KUOLLEENASYNT,TUPAKOINTITUNNUS,IVF,TROMBOOSIPROF,ANEMIA,SOKERI_PATOL,EI_LIEVITYSTA,EI_LIEVITYS_TIETOA,KAYNNISTYS,EDISTAMINEN,PUHKAISU,OKSITOSIINI,PROSTAGLANDIINI,ISTUKANIRROITUS,KAAVINTA,OMPELU,GBS_PROFYLAKSIA,AIDIN_ANTIBIOOTTIHOITO,VERENSIIRTO,YMPARILEIKKAUKSEN_AVAUS,KOHDUNPOISTO,EMBOLISAATIO,SYNNYTYSTAPATUNNUS,ETINEN,ISTIRTO,RKOURIS,HARTIADYSTOKIA,ASFYKSIA,PERATILA,SYNTYMATILATUNNUS,LAPSIVEDENMENO_PVM
end = time()
print("tmp file created in "+str(end-start)+" s")
```


```python
import csv
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)
```


```python
#based on the current understanding of the variables, they can be used as is in the final file
outname = "/data/projects/vaccination_project/data/vaccination_project_birthregistry_wide_082022_0fixed.csv"

start = time()
header_old = ['FINREGISTRYID','AVOLIITTO','KESKENMENOJA','KESKEYTYKSIA','ULKOPUOLISIA','KUOLLEENASYNT','SMOKE_NO','SMOKE_QUIT','SMOKE_YES','SMOKE_NA','IVF',
              'TROMBOOSIPROF','ANEMIA','SOKERI_PATOL','EI_LIEVITYSTA','EI_LIEVITYS_TIETOA','KAYNNISTYS','EDISTAMINEN','PUHKAISU','OKSITOSIINI','PROSTAGLANDIINI',
              'ISTUKANIRROITUS','KAAVINTA','OMPELU','GBS_PROFYLAKSIA','AIDIN_ANTIBIOOTTIHOITO','VERENSIIRTO','YMPARILEIKKAUKSEN_AVAUS','KOHDUNPOISTO','EMBOLISAATIO',
              'BIRTH_VAGINAL','BIRTH_BREECH','BIRTH_FORCEPS','BIRTH_VACUUM','BIRTH_PLANNEDC','BIRTH_URGENTC','BIRTH_EMERGENCYC','BIRTH_OTHERC','BIRTH_NA','ETINEN',
              'ISTIRTO','RKOURIS','HARTIADYSTOKIA','ASFYKSIA','PERATILA','BIRTHS_LIVE','BIRTHS_DIEDBEFORE','BIRTHS_DIEDDURING','BIRTHS_DIEDUNKNOWN','BIRTHS_NA']
header = ['FINREGISTRYID','BIRTH_COHABITING','BIRTH_MISCARRIAGES','BIRTH_TERMINATEDPREGNANCIES','BIRTH_TERMINATED_PREGNANCIES','BIRTH_STILLBORNS','BIRTH_SMOKE_NO',
          'BIRTH_SMOKE_QUIT','BIRTH_SMOKE_YES','BIRTH_SMOKE_NA','BIRTH_IVF','BIRTH_TROMBOSISPROFYLAXY','BIRTH_ANEMIA','BIRTH_GLUCOSETESTED','BIRTH_NOANALGESIA',
          'BIRTH_ANALGESIA_NA','BIRTH_ARTIFICIALLYINITIATED','BIRTH_PROMOTED','BIRTH_PUNCTURE','BIRTH_OXYTOCIN','BIRTH_PROSTAGLANDIN','BIRTH_MANUALEXTRACTION',
          'BIRTH_UTERINESCRAPING','BIRTH_SUTURING','BIRTH_GBSMED','BIRTH_MOTHERANTIBIOTICS','BIRTH_BLOODTRANSFUSION','BIRTH_CIRCUMCISION','BIRTH_HYSTERECTOMY',
          'BIRTH_EMBOLISATION','BIRTH_VAGINAL','BIRTH_BREECH','BIRTH_FORCEPS','BIRTH_VACUUM','BIRTH_PLANNEDC','BIRTH_URGENTC','BIRTH_EMERGENCYC','BIRTH_OTHERC',
          'BIRTH_NA','BIRTH_PLACENTAPRAEVIA','BIRTH_ABLATIOPLACENTAE','BIRTH_ECLAMPSIA','BIRTH_SHOULDERDYSTOCIA','BIRTH_ASPHYXIA','BIRTH_BREECH','BIRTH_S_LIVE',
          'BIRTH_S_DIEDBEFORE','BIRTH_S_DIEDDURING','BIRTH_S_DIEDUNKNOWN','BIRTH_S_NA','BIRTH_REG_NA']
birth_dict = {} #key = Mother's ID, value = list of values in order specified by header

with open(tmpfile,'rt') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        if row[0]=='AITI_TNRO':
            in_header = row
            continue
        ID = row[0]
        if ID not in study_ids: continue
        if ID not in birth_dict:
            birth_dict[ID] = [0 for i in range(len(header)-1)]
            birth_dict[ID][-1] = 0 #as this person is not missing from birth registry
        for i in list(range(2,7))+list(range(8,28))+list(range(29,35)):
            #These are the column indices of all variables that can be used as is from the original file
            wide_index = header_old.index(in_header[i])-1
            if birth_dict[ID][wide_index]==1: continue
            #print("i="+str(i))
            #print(row[i])
            #print(row)
            if len(row[i])<1: value = 9
            else: value = int(float(row[i]))
            if value==9 and birth_dict[ID][wide_index]==-1: birth_dict[ID][wide_index] = 'NA' 
            else: birth_dict[ID][wide_index] = value 
                        
        #smoking column is split into several variables
        smoking = int(row[7])
        if smoking==1: birth_dict[ID][header.index('BIRTH_SMOKE_NO')-1] = 1
        elif smoking==2: birth_dict[ID][header.index('BIRTH_SMOKE_QUIT')-1] = 1
        elif smoking==3 or smoking==4: birth_dict[ID][header.index('BIRTH_SMOKE_YES')-1] = 1
        else: birth_dict[ID][header.index('BIRTH_SMOKE_NA')-1] = 1
                
        #SYNNYTYSTAPATUNNUS is split into several variables
        if len(row[28])<1: mob = 9
        else: mob = int(float(row[28]))
        if mob==1: birth_dict[ID][header.index('BIRTH_VAGINAL')-1] = 1
        elif mob==2: birth_dict[ID][header.index('BIRTH_BREECH')-1] = 1
        elif mob==3: birth_dict[ID][header.index('BIRTH_FORCEPS')-1] = 1
        elif mob==4: birth_dict[ID][header.index('BIRTH_VACUUM')-1] = 1
        elif mob==5: birth_dict[ID][header.index('BIRTH_PLANNEDC')-1] = 1
        elif mob==6: birth_dict[ID][header.index('BIRTH_URGENTC')-1] = 1
        elif mob==7: birth_dict[ID][header.index('BIRTH_EMERGENCYC')-1] = 1
        elif mob==8: birth_dict[ID][header.index('BIRTH_OTHERC')-1] = 1
        else: birth_dict[ID][header.index('BIRTH_NA')-1] = 1
                    
        #SYNTYMATILATUNNUS is split into several variables
        if len(row[35])<1: status = 9
        else: status = int(float(row[35]))
        if status==1: birth_dict[ID][header.index('BIRTH_S_LIVE')-1] = 1
        elif status==2: birth_dict[ID][header.index('BIRTH_S_DIEDBEFORE')-1] = 1
        elif status==3: birth_dict[ID][header.index('BIRTH_S_DIEDDURING')-1] = 1
        elif status==4: birth_dict[ID][header.index('BIRTH_S_DIEDUNKNOWN')-1] = 1
        else: birth_dict[ID][header.index('BIRTH_S_NA')-1] = 1
                
        event_date = row[36]
        event_age = row[2]
            
end = time()
print("Data read in in "+str(end-start)+" s")

```


```python
missing_ids = study_ids-set(list(birth_dict.keys()))
print("Number of study population IDs missing from birth registry: "+str(len(missing_ids)))
```


```python
#write the wide birth registry file
start = time()
with open(outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(header)
    for ID in birth_dict: w.writerow([ID]+['NA' if value==-1 else value for value in birth_dict[ID]])#change -1s to NAs
    #then write entries for missing IDs
    for ID in missing_ids:
        row = ['NA' for i in range(len(header))]
        row[0] = ID
        row[-1] = 1
        w.writerow(row)
end = time()
print("Wide birth registry file written in "+str(end-start)+" s")
```

<a id='occupation_register'></a>
## Process occupation register

Columns to include in the final file:

1. FINREGISTRYID
2. OCCUPATION_CAT, first digit of the profession code (ammattikoodi), only 1995 classification or newer
3. OCCUPATION_ARMY, Armed forces (OCCUPATION_CAT=0)
4. OCCUPATION_MANAGER, Managers (OCCUPATION_CAT=1)
5. OCCUPATION_PROFESSIONAL, Professionals (OCCUPATION_CAT=2)
6. OCCUPATION_TECHNICIAN, TEchnicians and associate professionals (OCCUPATION_CAT=3)
7. OCCUPATION_CLERICAL, Clerical and support workers (OCCUPATION_CAT=4)
8. OCCUPATION_SERVICESALES, Service and sales workers (OCCUPATION_CAT=5)
9. OCCUPATION_AGRICULTURAL, Skilled agricultural, foresty and fishery workers (OCCUPATION_CAT=6)
10. OCCUPATION_CRAFT, Craft and related trades workers (OCCUPATION_CAT=7)
11. OCCUPATION_OPERATORS, Plant and machine operators, and assemplers (OCCUPATION_CAT=8)
12. OCCUPATION_ELEMENTARY, Elementary occupations (OCCUPATION_CAT=9)
13. OCCUPATION_UNKNOWN, Unknown (OCCUPATION_CAT=X)
14. OCCUPATION_NA, No data about occupation (all occupation predictors others to NA)

[Go to top of page](#top)


```python
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)

#read in the the preprocessed version of the minimal phenotype file to get the birth dates of individuals
mf_file = "/data/processed_data/minimal_phenotype/minimal_phenotype_2022-03-28.csv"#"/data/projects/vaccination_project/data/vaccination_project_minimalphenotype_012022.csv"
IDs = {} #key=FINREGISTRYID, value=year of birth
with open(mf_file,'rt') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        if row[0].strip('""')=='FINREGISTRYID': continue
        ID = row[0].strip('"')
        if ID not in study_ids: continue
        #print(row)
        #print(ID)
        birthyear = int(row[2].split('-')[0])
        #print(birthyear)
        IDs[ID] = birthyear
        #while True:
        #    z = input('any')
        #    break

print(list(IDs.keys())[0])
print("Number of individuals="+str(len(IDs.keys()))) 
```


```python
#read in the occupation variables
occupation_file = "/data/processed_data/sf_socioeconomic/ammatti_u1442_a.csv.finreg_IDsp"
outname = outname = '/data/projects/vaccination_project/data/vaccination_project_occupation_wide_082022.csv'

header = ['FINREGISTRYID','OCCUPATION_CAT','OCCUPATION_ARMY','OCCUPATION_MANAGER','OCCUPATION_PROFESSIONAL','OCCUPATION_TECHNICIAN',
          'OCCUPATION_CLERICAL','OCCUPATION_SERVICESALES','OCCUPATION_AGRICULTURAL','OCCUPATION_CRAFT','OCCUPATION_OPERATORS',
          'OCCUPATION_ELEMENTARY','OCCUPATION_UNKNOWN','OCCUPATION_NA']
occupation_dict = {'0':'OCCUPATION_ARMY','1':'OCCUPATION_MANAGER','2':'OCCUPATION_PROFESSIONAL',
                  '3':'OCCUPATION_TECHNICIAN','4':'OCCUPATION_CLERICAL','5':'OCCUPATION_SERVICESALES',
                  '6':'OCCUPATION_AGRICULTURAL','7':'OCCUPATION_CRAFT','8':'OCCUPATION_OPERATORS',
                  '9':'OCCUPATION_ELEMENTARY','X':'OCCUPATION_UNKNOWN'}

start = time()

all_IDs = set()
uniq_IDs_with_pamko = set() #set of unique IDs with a pamko code
uniq_IDs_with_ammattikoodi = set() #set of unique IDs with an ammattikoodi code
uniq_IDs_with_ammattikoodi_1995 = set() #set of unique IDs with a ammattikoodi code starting 1995
occupation_data = {} #key = ID, value = occupation code
with open(occupation_file,'rt',encoding='latin-1') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        ID = row[0].strip('""')
        if row[0].count('FINREGISTRYID'): continue
        if ID not in IDs: continue
        all_IDs.add(ID)
        pamko = row[2].strip('""')
        if len(pamko)>0: uniq_IDs_with_pamko.add(ID)
        code = row[3].strip('""')
        year = row[1].strip('""')
        if len(year)<1: year = 0
        else: year = int(float(year))
        if year<1995:
            if len(code)>0: uniq_IDs_with_ammattikoodi.add(ID)
            continue
        if len(code)<1: code = 'X'
        else:
            code = code[0]
            uniq_IDs_with_ammattikoodi.add(ID)
            uniq_IDs_with_ammattikoodi_1995.add(ID)
        if ID not in occupation_data: occupation_data[ID] = code #only use the last occupation info for each inidividual
        elif code!='X': occupation_data[ID] = code
end = time()
print("Occupation data read in in "+str(end-start)+" s")

missing_ids = study_ids-set(list(occupation_data.keys()))
print('Number of study IDs missing occupation data: '+str(len(missing_ids)))

```


```python
N = len(all_IDs)
print("Total number of IDs = "+str(N))
print("Total number of IDs with an ammattikoodi (after 1995) = "+str(len(uniq_IDs_with_ammattikoodi_1995))+" ("+str(100*len(uniq_IDs_with_ammattikoodi_1995)/N)+"% of all IDs)")
n = len(uniq_IDs_with_ammattikoodi.difference(uniq_IDs_with_ammattikoodi_1995))
print("Total number of IDs with ammattikoodi before 1995 but not after = "+str(n)+" ("+str(100*n/N)+"% of all IDs)")
m = len(uniq_IDs_with_pamko.difference(uniq_IDs_with_ammattikoodi_1995))
print("Total number of IDs with pamko but not ammattikoodi after 1995 = "+str(m)+" ("+str(100*m/N)+"% of all IDs)")
```


```python
import numpy as np
#plot age distribution of IDs that have only a pamko code
i = 0
for ID in IDs:
    i += 1
    if i>10: break
ages = []
for ID in uniq_IDs_with_pamko.difference(uniq_IDs_with_ammattikoodi_1995): ages.append(2021-IDs[ID])
print("Mean age with only pamko, no ammattikoodi: "+str(np.mean(ages))+"(std: "+str(np.std(ages))+", median age: "+str(np.median(ages))+")")

n,bins,patches = plt.hist(ages)
plt.xlabel('age')
plt.ylabel('count')
```


```python
#plot age distribution of IDs that have ammattikoodi after 1995
i = 0
for ID in IDs:
    i += 1
    if i>10: break
ages = []
for ID in uniq_IDs_with_ammattikoodi_1995: ages.append(2021-IDs[ID])
print("Mean age with ammattikoodi (1995-): "+str(np.mean(ages))+"(std: "+str(np.std(ages))+", median age: "+str(np.median(ages))+")")

n,bins,patches = plt.hist(ages)
plt.xlabel('age')
plt.ylabel('count')
```


```python
#plot the count of inidividuals per occupation category
data = {} #key = higher-level occupation code, value = count
for ID in occupation_data:
    if occupation_data[ID] not in data: data[occupation_data[ID]] = 1
    else: data[occupation_data[ID]] += 1

data = list(data.items())
data.sort(key=itemgetter(1),reverse=True)
fig,ax = plt.subplots()#figsize=fig_dims)
sns.barplot(x=[x[0] for x in data],y=np.array([x[1] for x in data])/np.sum([x[1] for x in data]),ax=ax)
ticks = plt.xticks()[0] 
plt.xticks(ticks=ticks,labels=[occupation_dict[x[0]] for x in data],rotation=90)
plt.ylabel('IDs per occupation category')
```


```python
#save the preprocessed occupation variables to a file
start = time()
with open(outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(header)
    for ID in occupation_data:
        new_row = [0 for i in range(len(header))]
        new_row[0] = ID
        new_row[1] = occupation_data[ID]
        new_row[header.index(occupation_dict[occupation_data[ID]])] = 1
        w.writerow(new_row)
    #then save entries for IDs missing from occupation register
    for ID in missing_ids:
        new_row = ['NA' for i in range(len(header))]
        new_row[0] = ID
        new_row[-1] = 1
        w.writerow(new_row)
end = time()
print("Writing the occupation variables into a file took "+str(end-start)+" s")
```

<a id='education'></a>
## Process registry of education


Columns to include in the final file:

- 1. FINREGISTRYID (hetu)
- 2. EDULEVEL_CAT (vuosi), first digit of the kaste_t2 column value, 1=education possibly ongoing
- 3. EDULEVEL_PREPRIMARY, Pre-primary education (EDULEVEL_CAT=0)
- 4. EDULEVEL_LOWERSECONDARY, Primary and lower secondary education (EDULEVEL_CAT=2)
- 5. EDULEVEL_UPPERSECONDARY, Upper secondary education (EDULEVEL_CAT=3)
- 6. EDULEVEL_SPECIALISTVOCATIONAL, Specialist vocational education (EDULEVEL_CAT=4)
- 7. EDULEVEL_SHORTCYCLETERTIARY, Short-cycle tertiary education (EDULEVEL_CAT=5)
- 8. EDULEVEL_1STCYCLEHIHGER, First-cycle higher education, e.g. Bachelor's (EDULEVEL_CAT=6)
- 9. EDULEVEL_2NDCYCLEHIGHER, Second-cycle higher education, e.g. Master's (EDULEVEL_CAT=7)
- 10. EDULEVEL_3RDCYCLEHIGHER, Third-cycle higher education, e.g. Doctor's (EDULEVEL_CAT=8)
- 11. EDULEVEL_NA, Level unknown or missing (EDULEVEL_CAT=9)
- 12. EDULEVEL_ONGOING, Education possibly ongoing (EDULEVEL_CAT=1), all individuals less than 35 years old
- 13. EDUFIELD_CAT, first two digits of the iscifi2013 column, 11=education possibly ongoing
- 14. EDUFIELD_GENERIC, Generic programmes and qualifications (EDUFIELD_CAT=00)
- 15. EDUFIELD_EDUCATION, Education (EDUFIELD_CAT=01)
- 16. EDUFIELD_ARTSHUM, Arts and humanities (EDUFIELD_CAT=02)
- 17. EDUFIELD_SOCIALSCIENCES, Social sciences, journalism and information (EDUFIELD_CAT=03)
- 18. EDUFIELD_BUSINESSADMINLAW, Business, administration and law (EDUFIELD_CAT=04)
- 19. EDUFIELD_SCIENCEMATHSTAT, Natural sciences, mathematics and statistics (EDUFIELD_CAT=05)
- 20. EDUFIELD_ICT, Information and communication technologies (EDUFIELD_CAT=06)
- 21. EDUFIELD_ENGINEERING, Engineering, manufacturing and construction (EDUFIELD_CAT=07)
- 22. EDUFIELD_AGRICULTURE, Agriculture, forestry, fisheries and veterinary (EDUFIELD_CAT=08)
- 23. EDUFIELD_HEALTH, Health and wellfare (EDUFIELD_CAT=09)
- 24. EDUFIELD_SERVICES, Services (EDUFIELD_CAT=10)
- 25. EDUFIELD_NA, Field of education not found or unknown (EDUFIELD_CAT=99)
- 26. EDUFIELD_ONGOING, education possibly ongoig meaning person is less than 35 years old (EDUFIELD_CAT=11), 


Note that the unique values for the highest degree are 3-8, although by definition we could have 0-9. Variable "education can still be ongoing" is set for each individual aged between 30 and 35. Education level variables have been created by taking the first digit of the column kaste_t2. Naming is based on:  https://www2.stat.fi/en/luokitukset/koulutusaste/koulutusaste_1_20160101/

The field of education variables are derived by taking the first two digits of column iscifi2013 (these are ISCED 2013 codes). Note that only individuals who are born before 1992 are included in this analysis (these are the only individuals that are included in the preprocessed temporary minimal phenotype file created for this project).

Also, individuas between 30-35 years old are treated so that their education can still be ongoing, meaning that for all of them we set EDUFIELD_ONGOING=1.

For each individual, the information from the latest entry is used, unless the newest education level is lower than a previous one.

[Go to top of page](#top)


```python
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)

#read in the the preprocessed version of the minimal phenotype file to get the birth dates of individuals
mf_file = "/data/processed_data/minimal_phenotype/minimal_phenotype_2022-03-28.csv"
IDs = {} #key=FINREGISTRYID, value=year of birth
with open(mf_file,'rt') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        if row[0].strip('""')=='FINREGISTRYID': continue
        ID = row[0].strip('"')
        if ID not in study_ids: continue
        birthyear = int(row[2].split('-')[0])
        IDs[ID] = birthyear
                
print("Number of study population individuals="+str(len(IDs.keys()))) 
```


```python
#read in the education variables
education_file = "/data/processed_data/sf_socioeconomic/tutkinto_u1442_a.csv.finreg_IDsp"
outname = '/data/projects/vaccination_project/data/vaccination_project_education_wide_082022.csv'

#header = ['FINNREGISTRYID','HIGHEST_DEGREE_0','HIGHEST_DEGREE_3','HIGHEST_DEGREE_4','HIGHEST_DEGREE_5','HIGHEST_DEGREE_6','HIGHEST_DEGREE_7','HIGHEST_DEGREE_8','HIGHEST_DEGREE_-1','EDU_FIELD_00','EDU_FIELD_01','EDU_FIELD_02','EDU_FIELD_03','EDU_FIELD_04','EDU_FIELD_05','EDU_FIELD_06','EDU_FIELD_07','EDU_FIELD_08','EDU_FIELD_09','EDU_FIELD_10','EDU_FIELD_99','EDU_FIELD_NA','EDU_FIELD_OG']
header = ['FINREGISTRYID','EDULEVEL_CAT','EDULEVEL_PREPRIMARY','EDULEVEL_LOWERSECONDARY','EDULEVEL_UPPERSECONDARY','EDULEVEL_SPECIALISTVOCATIONAL','EDULEVEL_SHORTCYCLETERTIARY','EDULEVEL_1STCYCLEHIGHER','EDULEVEL_2NDCYCLEHIGHER',
          'EDULEVEL_3RDCYCLEHIGHER','EDULEVEL_NA','EDULEVEL_ONGOING','EDUFIELD_CAT','EDUFIELD_GENERIC','EDUFIELD_EDUCATION','EDUFIELD_ARTSHUM','EDUFIELD_SOCIALSCIENCES','EDUFIELD_BUSINESSADMINLAW','EDUFIELD_SCIENCEMATHSTAT',
          'EDUFIELD_ICT','EDUFIELD_ENGINEERING','EDUFIELD_AGRICULTURE','EDUFIELD_HEALTH','EDUFIELD_SERVICES','EDUFIELD_NA','EDUFIELD_ONGOING']

#dictionaries for converting numerical codes to variable names
eduleveldict = {'0':'EDULEVEL_PREPRIMARY','1':'EDULEVEL_ONGOING','2':'EDULEVEL_LOWERSECONDARY','3':'EDULEVEL_UPPERSECONDARY','4':'EDULEVEL_SPECIALISTVOCATIONAL',
'5':'EDULEVEL_SHORTCYCLETERTIARY','6':'EDULEVEL_1STCYCLEHIGHER','7':'EDULEVEL_2NDCYCLEHIGHER','8':'EDULEVEL_3RDCYCLEHIGHER','9':'EDULEVEL_NA'}

edufielddict = {'00':'EDUFIELD_GENERIC','01':'EDUFIELD_EDUCATION','02':'EDUFIELD_ARTSHUM',
'03':'EDUFIELD_SOCIALSCIENCES','04':'EDUFIELD_BUSINESSADMINLAW','05':'EDUFIELD_SCIENCEMATHSTAT',
'06':'EDUFIELD_ICT','07':'EDUFIELD_ENGINEERING','08':'EDUFIELD_AGRICULTURE','09':'EDUFIELD_HEALTH',
'10':'EDUFIELD_SERVICES','99':'EDUFIELD_NA','11':'EDUFIELD_ONGOING'}

edulevel_cat_index = header.index('EDULEVEL_CAT')-1
edufield_cat_index = header.index('EDUFIELD_CAT')-1

start = time()
edu_dict = {}

Ncap = 100000
unknown_edulevel_descriptions = set() #This is for trying to figure out the meaning of the edu level codies starting with 4

with open(education_file,'rt',encoding='latin-1') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        if row[0].strip('""')=='FINREGISTRYID': continue
        ID = row[0].strip().strip('""')
        #if individual is not in the stud population, they are skipped
        if ID not in IDs: continue
        if IDs[ID]>1987:
            edu_dict[ID] = [0 for i in range(len(header)-1)] #Only the latest entry is used for each individual
            edu_dict[ID][header.index('EDUFIELD_ONGOING')-1] = 1
            edu_dict[ID][header.index('EDULEVEL_ONGOING')-1] = 1
            #save the values of the categorical variales
            edu_dict[ID][edulevel_cat_index] = '1'
            edu_dict[ID][edufield_cat_index] = '11'
        else:    
            edu_field = row[3].strip().strip('"')[:2]
            #print("EDU_FIELD:"+edu_field)
            if len(edu_field)<1: edu_field = '99'
            edu_level = row[4].strip().strip('""')[:1]
            if len(edu_level)<1: edu_level = '9' #99 marks NA
            #if edu_level!='9': print(row)
            edu_field_index = header.index(edufielddict[edu_field])-1
            if edu_level=='4':
                #edu_level_index = header.index('SES_NAEDU')-1
                unknown_edulevel_descriptions.add(row[5].strip('""'))
                
            edu_level_index = header.index(eduleveldict[edu_level])-1
            if ID not in edu_dict:
                edu_dict[ID] = [0 for i in range(len(header)-1)]
                edu_dict[ID][edu_level_index] = 1
                edu_dict[ID][edu_field_index] = 1
                #save the values of the categorical variales
                edu_dict[ID][edulevel_cat_index] = edu_level
                edu_dict[ID][edufield_cat_index] = edu_field
            else:
                if edu_level!='9':
                    if (edu_dict[ID][edulevel_cat_index]=='9') or (int(edu_level)>int(edu_dict[ID][edulevel_cat_index])):
                        #the newer entry is only kept if education level is higher than before
                        edu_dict[ID] = [0 for i in range(len(header)-1)]
                        edu_dict[ID][edu_level_index] = 1
                        edu_dict[ID][edu_field_index] = 1
                        #save the values of the categorical variales
                        edu_dict[ID][edulevel_cat_index] = edu_level
                        edu_dict[ID][edufield_cat_index] = edu_field
       
print("Number of IDs="+str(len(list(edu_dict.keys()))))
end = time()
print("Education variables read in in "+str(end-start)+" s")
```


```python
#Plot the distributions of the categorical variables
#first level of education
data = {} #key = education level code, value = count
for ID in edu_dict:
    #print(edu_dict[ID])
    if edu_dict[ID][edulevel_cat_index] not in data: data[edu_dict[ID][edulevel_cat_index]] = 1
    else: data[edu_dict[ID][edulevel_cat_index]] += 1

data = list(data.items())
data.sort(key=itemgetter(1),reverse=True)
print(data)
fig,ax = plt.subplots()#figsize=fig_dims)
sns.barplot(x=[x[0] for x in data],y=[x[1] for x in data],ax=ax)
ticks = plt.xticks()[0]
plt.xticks(ticks=ticks,labels=[eduleveldict[x[0]] for x in data],rotation=90)
plt.ylabel('IDs per education level')
plt.xlabel('education level')
```


```python
#Then field of education
data = {} #key = field of education code, value = count
for ID in edu_dict:
    if edu_dict[ID][edufield_cat_index] not in data: data[edu_dict[ID][edufield_cat_index]] = 1
    else: data[edu_dict[ID][edufield_cat_index]] += 1

data = list(data.items())
data.sort(key=itemgetter(1),reverse=True)
fig,ax = plt.subplots()#figsize=fig_dims)
sns.barplot(x=[x[0] for x in data],y=[x[1] for x in data],ax=ax)
ticks = plt.xticks()[0]
plt.xticks(ticks=ticks,labels=[edufielddict[x[0]] for x in data],rotation=90)
plt.ylabel('IDs per field of education')
plt.xlabel('field of education')
```


```python
#get the number of IDs missing education info
missing_ids = study_ids-set(list(edu_dict.keys()))
print('Number of study population IDs missing education data: '+str(len(missing_ids)))
```


```python
#save the preprocessed education variables to a file
start = time()
with open(outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(header)
    for ID in edu_dict: w.writerow([ID]+edu_dict[ID])
    #save entries for missing IDs
    for ID in missing_ids:
        new_row = [0 for i in range(len(header))]
        new_row[0] = ID
        new_row[1] = 9
        new_row[10] = 1
        new_row[12] = 99
        new_row[24] = 1
        w.writerow(new_row)
end = time()
print("Writing the education variables into a file took "+str(end-start)+" s")
```

<a id='update_drug'></a>
## Update the drug purchase file variable names to more interpretable names and truncate ATC-codes to first 5 digits

The variable names are changed to DRUG_NAME_ATC, where NAME is the name corresponding to the ATC code as defined in file: xxx.

ATC-codes are truncated to first 5 characters to reduce the number of correlating variables.

[Go to top of page](#top)


```python
import numpy as np
import csv
from time import time
inname = "/data/projects/vaccination_project/data/drug_purchases_binary_wide_ALL.csv"
outname = "/data/projects/vaccination_project/data/drug_purchases_binary_wide_ALL_newnames_082022.csv"
ATC_to_namefile = "/data/projects/vaccination_project/data/atc_codes_wikipedia.csv"

#first read in the ATC code to drug name mapping
ATC_to_name = {} #key = ATC, value = drug name with whitespace replaced with _
with open(ATC_to_namefile,'rt') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        if row[0]=='code': continue
        ATC_to_name[row[0]] = row[1].replace(' ','_')
        
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)
```


```python
start = time()
#read in the header of the current drug purchase file
drug_IDs = []
with open(inname,'rt') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        if row[0]=='FINREGISTRYID':
            #update the header
            header = row
        else: drug_IDs.append(row[0])
print(header)
print("Number of IDs="+str(len(drug_IDs)))
#read in the drug purchase data
drug_purchases = np.loadtxt(inname,delimiter=',',skiprows=1,usecols=[i for i in range(1,len(header))])
end = time()
print("Drug purchase data read in in "+str(end-start)+" s")
```


```python
#get the list of truncated ATC code names
trunc_ATCs = set()
for ATC in header[1:]: trunc_ATCs.add(ATC[:5])
trunc_ATCs = list(trunc_ATCs)
#update the header
new_header = ['FINREGISTRYID']
for ATC in trunc_ATCs:
    if ATC not in ATC_to_name: newname = 'DRUG_NONAME_'+ATC
    else: newname = 'DRUG_'+ATC_to_name[ATC]+"_"+ATC
    new_header.append(newname) 
```


```python
start = time()
#create a new truncated drug matrix
drug_purchases_trunc = np.zeros(shape=(drug_purchases.shape[0],len(trunc_ATCs)))
i = 0
for trunc_ATC in trunc_ATCs:
    #get all longer ATC codes that match the truncated code
    ATC_inds = [header.index(ATC) for ATC in header[1:] if ATC[:5]==trunc_ATC]
    for ind in ATC_inds: drug_purchases_trunc[:,i] += drug_purchases[:,ind-1]
    i += 1
#binarize the matrix
drug_purchases_trunc = np.where(drug_purchases_trunc>0,1,0)
end = time()
print("Creating the truncated drug purchase matrix took "+str(end-start)+" s")
```


```python
start = time()
#save the updated drug purchases to a file
missing_ids = study_ids - set(drug_IDs)
print('Number of study population IDs not having drug purchase data: '+str(len(missing_ids)))
with open(outname,'w') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(new_header)
    for i in range(0,len(drug_IDs)):
        if drug_IDs[i] not in study_ids: continue
        w.writerow([drug_IDs[i]]+list(drug_purchases_trunc[i,:]))
    for ID in missing_ids: w.writerow([ID]+[0 for i in range(len(new_header)-1)])
end = time()
print("Truncated drug purchase file written in "+str(end-start)+" s")
```

<a id='update_endpoint'></a>
## Update endpoint variable names to more interpretable names

The variable names are DISEASE_DISEASENAME, as taken from Finngen endpoint definitions in: FINNGEN_ENDPOINTS_DF8_Final_2021-09-02.csv

We also remove some of the rare and redundant endpoints following the process defined by Andrius. The endpoints to keep have been saved into the file /data/projects/vaccination_project/data/keep_endpoints_from_Andrius.csv

[Go to top of page](#top)


```python
import csv
from time import time

start = time()
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)
        
#read in the preprocessed endpoint file and filter out IDs that are not in the study population
inname = "/data/projects/vaccination_project/data/wide_first_events_endpoints_dicot.csv"
inname_filtered = "/data/projects/vaccination_project/data/wide_first_events_endpoints_dicot_filtered.csv"
endpoint_ids = set()

with open(inname_filtered,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    with open(inname,'rt') as infile:
        r = csv.reader(infile,delimiter=',')
        for row in r:
            if row[0].count('FINREGISTRY'):
                w.writerow(row)
                old_header = row
            else:
                ID = row[0]
                if ID in study_ids:
                    w.writerow(row)
                    endpoint_ids.add(ID)
        #add rows for IDs that have missing endpoint data, here, all predictors are just given value 0
        missing_ids = study_ids-endpoint_ids
        print('Number of study population IDs missing from endpoint data: '+str(len(missing_ids)))
        for ID in missing_ids: w.writerow([ID]+[0 for i in range(len(old_header)-1)])
end = time()
print('Filtering the endpoint data took '+str(end-start)+" s")
```


```python
import pandas as pd

truncatedname = "/data/projects/vaccination_project/data/wide_first_events_endpoints_dicot_truncated.csv"
outname = "/data/projects/vaccination_project/data/wide_first_events_endpoints_dicot_newnames_08S2022.csv"
keep_endpoints = '/data/projects/vaccination_project/data/keep_endpoints_from_Andrius_new.csv'
endpoint_to_namefile = "/data/projects/vaccination_project/data/FINNGEN_ENDPOINTS_DF8_Final_2021-09-02.csv"

#read in the names of columns from the endpoint file
with open(inname_filtered,'rt') as infile:
    for row in infile:
        header = row.strip('\n').split(',')
        break
 
print("Original number of columns="+str(len(header)))
#then read in the names of endpoints to keep
keeps = []
with open(keep_endpoints,'rt') as infile:
    for row in infile:
        keeps = keeps+row.strip('\n').split(',')
        break
#add _NEVT to end of each kept variable
for i in range(0,len(keeps)): keeps[i] = keeps[i]+'_NEVT'
keep_cols = list(set(keeps).intersection(set(header)))
print("Number of columns to keep="+str(len(keep_cols)))
```


```python
#read in the endpoint data in chunks and save to a file
df_iterator = pd.read_csv(inname_filtered,usecols=['FINREGISTRYID']+keep_cols,sep=',',chunksize=500000)

for i, df_chunk in enumerate(df_iterator):
    mode = 'w' if i==0 else 'a'
    header = i==0
    
    df_chunk.to_csv(truncatedname,index=False,header=header,mode=mode)
    print(str(500000*(i+1))+" rows processed.")
```


```python
import csv
#replace the header of the endpoint file with the updated variable names
#first read in the endpoint to long name mapping
endpoint_to_name = {} #key = endpoint name, value = long name with whitespace replaced with _
with open(endpoint_to_namefile,'rt') as infile:
    for row in infile:
        row = row.strip('\n').split(',')
        if row[0]=='TAGS': continue
        endpoint_to_name[row[3]] = row[4].replace(' ','_')
#first read in the current header
newheaderfile = "/data/projects/vaccination_project/data/wide_first_events_endpoints_dicot_newheader.csv"
with open(newheaderfile,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    with open(truncatedname,'rt') as infile:
        for row in infile:
            row = row.strip('\n').split(',')
            print(row[:10])
            print(len(row))
            new_header = [row[0]]
            for i in range(1,len(row)):
                endpoint = row[i]
                endpoint = endpoint[:endpoint.rindex('_')]
                if row[i] in endpoint_to_name: new_header.append('DISEASE_'+endpoint_to_name[endpoint])
                else: new_header.append('DISEASE_'+endpoint)
            w.writerow(new_header)
            break
print(new_header[:10]) 
print(len(new_header))
```


```python
#replace the header
from os import system
start = time()
system('(cat '+newheaderfile+'; tail -n+2 '+truncatedname+') > '+outname)
end = time()
print('Header replaced in '+str(end-start)+' s')
```

<a id='relative_vax'></a>
## Create variables describing vaccination status of relatives

The ouput intermediate file contains the following columns:

- 1. FINREGISTRYID
- 2. REL_ISMOTHERVACC, =0 if mother is vaccinated, =1 if mother is not vaccinated, =NA if no information about mother's vaccination status is available
- 3. REL_ISFATHERVACC, =0 if father is vaccinated, =1 if fater is not vaccinated, =NA if no information about father's vaccination status is available
- 4. REL_ISSIBLINGVACC, =0 if any sibling is vaccinated, =1 if no siblings are vaccinated, =NA if no information about siblings' vaccination status is available
- 5. REL_ISMOTHERNA, =1 if mother is missing, otherwise 0
- 6. REL_ISFATHERNA, =1 if father is missing, otherwise 0
- 7. ISSIBLINGNA, =1 if no siblings in the study population, otherwise 0

Information about relatives for each person is retrieved from DVV relatives. If the relative is not included in the study population, the value of the corresponding REL-variable is NA. There are several reasons why a person's relative would not be included in the study population. They can be too young (<30 yo), too old (>80 yo), dead, emigrated... 

[Go to top of page](#top)


```python
import pandas as pd
#read in the DVV relatives
rel_filename = "/data/processed_data/dvv/Tulokset_1900-2010_tutkhenk_ja_sukulaiset.txt.finreg_IDsp"
df_rel = pd.read_csv(rel_filename,usecols=['FINREGISTRYID','Relationship','Relative_ID'])
df_rel
```


```python
#read in the vaccination status
vac_status_name = "/data/projects/vaccination_project/data/vaccination_outcome_including_covid19+_052022.csv"
df_vacc = pd.read_csv(vac_status_name,usecols=['FINREGISTRYID','COVIDVax'])
df_vacc
```


```python
#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)
```


```python
df_vacc_dict = df_vacc.to_dict(orient='records')
df_vacc_dict2 = {} #key = FINREGISTRYID, value = COVIDVax
for l in df_vacc_dict:
    df_vacc_dict2[l['FINREGISTRYID']] = l['COVIDVax']
IDs_with_vacc_info = set(df_vacc_dict2.keys()) #for faster testing
```


```python
#Add mother's ID as a column
df_vacc = df_vacc.merge(df_rel.loc[df_rel['Relationship']=='3a'],how='left',left_on='FINREGISTRYID',right_on='FINREGISTRYID')
df_vacc = df_vacc.rename(columns={'Relationship':'Mother','Relative_ID':'Mother_ID'})
df_vacc
```


```python
#Add Father's ID as a column
df_vacc = df_vacc.merge(df_rel.loc[df_rel['Relationship']=='3i'],how='left',left_on='FINREGISTRYID',right_on='FINREGISTRYID')
df_vacc = df_vacc.rename(columns={'Relationship':'Father','Relative_ID':'Father_ID'})
df_vacc
```


```python
#Add indicator column describing whether the individual is in the study population or not
study_pop_column = []
for index,row in df_vacc.iterrows():
    #print(row)
    ID = row['FINREGISTRYID']
    if ID in study_ids: study_pop_column.append(1)
    else: study_pop_column.append(0)
df_vacc['In study population'] = study_pop_column
df_vacc
```


```python
#subset to study population only
df_vacc_study_pop = df_vacc.loc[df_vacc['In study population']>0]
df_vacc_study_pop
```

```python
from time import time
start = time()
#subset df_rel to siblings only for faster searching
df_rel_sibs = df_rel.loc[df_rel['Relationship'].isin(['4i','4a'])]
df_rel_sibs = df_rel_sibs.loc[df_rel_sibs['FINREGISTRYID'].isin(study_ids)]
dict_rel_sibs = df_rel_sibs.to_dict(orient='records')
end = time()
print('DIctionary containing only siblings created in '+str(end-start)+" s")
len(df_rel_sibs['FINREGISTRYID'].unique())
```


```python
start = time()
dict_rel_sibs2 = {} #key = ID, value = list of sibling IDs
for l in dict_rel_sibs:
    ID = l['FINREGISTRYID']
    sib_ID = l['Relative_ID']
    if sib_ID not in study_ids: continue
    if ID not in dict_rel_sibs2: dict_rel_sibs2[ID] = [sib_ID]
    else: dict_rel_sibs2[ID].append(sib_ID)
    #if sib_ID not in df_vacc_dict: dict_rel_sibs2[ID] = 'NA' 
    #elif ID not in dict_rel_sibs2: dict_rel_sibs2[ID] = df_vacc_dict2l[sib_ID]
    #else: dict_rel_sibs2[ID] = min([dict_rel_sibs2[ID],df_vacc_dict2[sib_ID]])
end = time()
print("Sibling dictionary reformatted in "+str(end-start)+" s")
len(dict_rel_sibs2.keys())        
```


```python
#write the results row by row
import csv
from time import time
#read in each individual ID  and add the vaccination status of relatives
start = time()
outname = "/data/projects/vaccination_project/data/vaccination_project_relative_vaccination_status_082022.csv"
test_counter = 0
test_max = 1000
rel_ids = set()

with open(outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    header = ['FINREGISTRYID','REL_ISMOTHERVACC','REL_ISFATHERVACC','REL_ISSIBLINGVACC','REL_ISMOTHERNA','REL_ISFATHER_NA','REL_ISSIBLINGNA']
    w.writerow(header)
    print("Header written...")
    for index,row in df_vacc_study_pop.iterrows():
        ID = row['FINREGISTRYID']
        if ID in rel_ids: continue
        rel_ids.add(ID)
        new_row = [ID,'NA','NA','NA',1,1,1]
        Mother_ID = row['Mother_ID']
        if Mother_ID in study_ids:
            new_row[1] = df_vacc_dict2[Mother_ID]
            new_row[4] = 0
        Father_ID = row['Father_ID']
        if Father_ID in study_ids: 
            new_row[2] = df_vacc_dict2[Father_ID]
            new_row[5] = 0
        
        if ID not in dict_rel_sibs2:
            w.writerow(new_row)
            continue
        for sib_ID in dict_rel_sibs2[ID]:
            vacc_status = df_vacc_dict2[sib_ID]
            new_row[3] = vacc_status
            new_row[6] = 0
            if vacc_status==0: break
        w.writerow(new_row)
    #then add entries for missing IDs
    missing_ids = study_ids-rel_ids
    print('Number of study population IDs not having any info about relatives: '+str(len(missing_ids)))
    for ID in missing_ids: w.writerow([ID,'NA','NA','NA',1,1,1])
end = time()
print("Variables measuring vaccination status of relatives written in "+str(end-start)+" s")  
```

<a id='vacc_status'></a>
## Filter the vaccination status file

Remove not needed variables and IDs that are not in the study population.

[Go to top of page](#top)


```python
import pandas as pd
import csv

#read in the study population ids
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"
with open(idfile,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        study_ids = set(row)
        
vacc_status_name = "/data/projects/vaccination_project/data/vaccination_outcome_including_covid19+_052022.csv"
outname = "/data/projects/vaccination_project/data/vaccination_outcome_082022.csv"

df = pd.read_csv(vacc_status_name,delimiter=',',usecols=['FINREGISTRYID','age_october_2021','first_visit','COVIDVax'])
df
```


```python
#remove IDs not in study population
include_col = []
for index,row in df.iterrows():
    ID = row['FINREGISTRYID']
    if ID in study_ids: include_col.append(1)
    else: include_col.append(0)
df['include'] = include_col
df_filtered = df.loc[df['include']>0]
df_filtered
```


```python
#save the resulting df to a file
df_filtered.to_csv(outname,index=False,sep=',',columns=['FINREGISTRYID','age_october_2021','first_visit','COVIDVax'])
```

<a id='relative_vax'></a>
## Combine the individual files into one training and one test set file

Note that each of the files has been sorted by the ID, and each contains exactly the same IDs, sorting example:

`sort -t ',' -k1,1 /data/projects/vaccination_project/data/vaccination_project_infectious_diseases_082022.csv > /data/projects/vaccination_project/data/vaccination_project_infectious_diseases_082022.csv.sorted`

[Go to top of page](#top)


```python

```


```python
import pandas as pd
from time import time
path = '/data/projects/vaccination_project/data/'
sorted_files = ['wide_first_events_endpoints_dicot_newnames_08S2022.csv.sorted','drug_purchases_binary_wide_ALL_newnames_082022.csv.sorted',
                'vaccination_project_marriage_082022.csv.sorted',
                'vaccination_project_socialassistanceregister_wide_082022.csv.sorted','vaccination_project_birthregistry_wide_082022_0fixed.csv',
                'vaccination_project_minimalphenotype_DVV_mt_082022.csv.sorted','vaccination_project_soshilmo_wide_082022.csv.sorted',
                'vaccination_project_education_wide_082022.csv.sorted','vaccination_project_occupation_wide_082022.csv.sorted',
                'vaccination_project_yearly_earnings_082022.csv.sorted','vaccination_project_infectious_diseases_082022.csv.sorted',
                'vaccination_project_relative_vaccination_status_082022.csv.sorted',
               'vaccination_outcome_082022.csv.sorted']
```


```python
#read in the first file
start = time()

df = pd.read_csv(path+sorted_files[0],delimiter=',')
end = time()
print("First file read in in "+str(end-start)+' s')
df
```


```python
#merge the other files
start = time()
for fname in sorted_files[1:]:
    df = pd.merge(df,pd.read_csv(path+fname,delimiter=','),on='FINREGISTRYID')
    print(fname+' merged, number of columns is '+str(len(df.columns)))
end = time()
print("Merging done in "+str(end-start)+' s')
df
```


```python
out_name = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022.csv'
df.to_csv(out_name,sep=',',index=False)
```


```python
#remove IDs corresponding to people living in Askola
#NOTE: this can only be done afer running the vacc_stats.ipynb notebook

askola_ids_file = '/data/projects/vaccination_project/data/vaccination_project_study_ids_living_in_Askola_082022.csv'
out_name_filtered = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_no_Askola.csv'
from os import system
from time import time
start = time()
system('grep -v -f '+askola_ids_file+' '+out_name+' > '+out_name_filtered)
end = time()
print('People living in Askola filtered out in '+str(end-start)+' s')
```


```python
#read in the IDs of people living in Askola
import csv
with open(askola_ids_file,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    askola_ids = []
    for row in infile: askola_ids.append(row.strip())
```


```python
import random
import csv
#split to training and test sets and save these
train_name = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_train.csv'
test_name = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_test.csv'

askola_ids = set(askola_ids)

#assign 80% of individuals to training, 20% to test set
random.seed(42)
indices = set(list(df.index))
train_inds = set(random.sample(indices,int(0.8*len(df))))
test_inds = indices - train_inds

header = list(df.columns)
with open(train_name,'wt') as train_file:
    w_train = csv.writer(train_file,delimiter=',')
    w_train.writerow(header)
    with open(test_name,'wt') as test_file:
        w_test = csv.writer(test_file,delimiter=',')
        w_test.writerow(header)

        ind = 0
        for index in df.index:
            if df.iloc[index]['FINREGISTRYID'] in askola_ids: continue #skip individuals living in Askola
            if index in test_inds: w_test.writerow(list(df.iloc[index]))
            else: w_train.writerow(list(df.iloc[index]))
            ind += 1
            if ind%500000<1: print(str(ind)+" rows written.")
    
```


```python
#save to file
df_test.to_csv(test_name,sep=',')
del_df_test
```


```python
#then the training set
df_train = df.iloc[train_inds]
df_train.to_csv(train_name,sep=',')
```

<a id='create_input'></a>
## Create input variable lists for each logistic regression model

[Go to top of page](#top)


```python
import re
import csv

test_name = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_test.csv'

#read in the header
with open(test_name,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        header = row
        break

omits = ['FINREGISTRYID', 'first_visit', 'last_visit', 'difference', 'doseno56','DISEASE_DEATH']
            
#always omit the following municipalities

omit_munis = ['Askola']

for o in omit_munis: omits.append('GEO_'+o)
for name in header:
    if name.count('_CAT')>0: omits.append(name) #categorical variables are handled separately
```


```python
#remove extra ", ( and ) characters from column names
#replace /, \ and - characters with _
#and truncate column names to max 400 characters
for i in range(len(header)):
    newname = header[i].replace('"','').replace('(','').replace(')','').replace('/','_').replace('\\','_').replace('-','_').replace(' ','_')
    if len(newname)>400: newname = newname[:400]
    header[i] = newname
print(header[:10])   
```


```python
#save the new header to file and replace the headers of the old training and test files with the new header
newheader_name = '/data/projects/vaccination_project/data/vaccination_project_newheader_27082022.csv'
with open(newheader_name,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(header)
    
#replace the headers of files used in model training:
full_name = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_no_Askola.csv'
full_train_name = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_train.csv'
full_test_name= '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_test.csv'
imputed_test_name = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_test_imputed.csv'
files = [imputed_test_name]#[full_name,full_train_name,full_test_name]#[downsampled_train_name,downsampled_imputed_train_name,imputed_test_name]


from os import system
for file in files:
    cmd = '( head -1 '+newheader_name+'; tail -n +2 '+file+' ) > '+file+'.newheader'
    print(cmd)
    system(cmd)
print('done!')
```


```python
#read in the endpoints that should be kept
with open('/data/projects/vaccination_project/data/keep_endpoints_from_Andrius_new.csv','rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        keep_endpoints = ['DISEASE_'+r for r in row]
        break
print(keep_endpoints[:10])
for name in header:
    if name.count('DISEASE_')>0 and name not in keep_endpoints:
        omits.append(name)
        
#print(omits)
#add to omits the following non-binary variables from irth registry:
omits += ['BIRTH_COHABITING','BIRTH_MISCARRIAGES','BIRTH_TERMINATED_PREGNANCIES','BIRTH_TERMINATEDPREGNANCIES','BIRTH_STILLBORNS',
         'BIRTH_NOANALGESIA','BIRTH_PUNCTURE','BIRTH_PLACENTAPRAEVIA']
print(header)
print('--------------------')
print(len(omits))
print('------------------')
print(omits[:10])
#save to a file
omitcolumnfile = '/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_082022_binary_omit_columns.csv'
with open(omitcolumnfile,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(omits)
```


```python
from time import time
#create N files that can be used to run the logistic regression script in parallel for all binary variables
start = time()
N = 10
filenames = []
for i in range(N): filenames.append('/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_082022_fit_binary_variables_'+str(i)+'.csv')
always_include = set(["COVIDVax","SEX","age_october_2021",'GEO_Helsinki','MOTHERTONGUE_Finnish','SES_LOWERLEVEL','EDULEVEL_UPPERSECONDARY','OCCUPATION_SERVICESALES',
                      'EARNINGS_5DEC']) #these are needed in all files
omit_set = set(omits)
all_set = set(header)
fit_vars = list(all_set-omit_set-always_include)
print("Number of variables to fit = "+str(len(fit_vars)))
L = int(len(fit_vars)/N)
print('L='+str(L))

var_splits = []
for i in range(0,len(fit_vars),L): var_splits.append(fit_vars[i:i+L])
var_splits[-1] = fit_vars[i:]
    
i = 0
for filename in filenames:
    with open(filename,'wt') as outfile:
        w = csv.writer(outfile,delimiter=',')
        w.writerow(list(always_include)+var_splits[i])
    i += 1
end = time()
print("Binary variable splits done in "+str(end-start)+" s")
```


```python
#save individual input files, one per each binary variable
start = time()
outdir = '/data/projects/vaccination_project/data/logreg_input_features/082022_'
with open(outdir+'all_fit_varnames.txt','wt') as outfile:
    w_all = csv.writer(outfile,delimiter='\t')
    for var in always_include: w_all.writerow([var])
    for var in fit_vars:
        #if var!='MOTHERTONGUE_NOFINSWE': continue
        w_all.writerow([var])
        if var.count('/')<1:
            with open(outdir+var+'_variables.csv','wt') as outfile:
                w = csv.writer(outfile,delimiter=',')
                w.writerow(list(always_include)+[var])
end = time()
print("Individual input files written in "+str(end-start)+" s")

with open(outdir+'all_fit_varnames_onerow.txt','wt') as outfile:
        w_all = csv.writer(outfile,delimiter=',')
        w_all.writerow(list(always_include)+fit_vars)
```


```python
from time import time
#create similar files including all needed "dummy" binary variables for each catecorigal variable
outdir = '/data/projects/vaccination_project/data/logreg_input_features/082022_'

always_include = set(["COVIDVax","SEX","age_october_2021"]) #these are needed in all files

#dictionary containing the lists of predictors per each category
cat_variables = {}#key = category name, value = list of predictors in the category
all_vars = [] #this list contains all variables that are in any of the categories (including the baseline variables)
all_vars += list(always_include)
start = time()

#RELATIVES_CAT
catvar = "RELATIVES_CAT"
vars = ["REL_ISMOTHERVACC","REL_ISFATHERVACC","REL_ISSIBLINGVACC",'REL_ISMOTHERNA','REL_ISFATHER_NA','REL_ISSIBLINGNA']
all_vars += vars
cat_variables[catvar] = vars
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)

#ZIPCODE_ZIPCODE_CAT
catvar = "GEO_MUNICIPALITY_CAT"
vars = []
for name in header:
    if name.count('GEO_')>0 and name[-3:]!='CAT' and name not in omits: vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)
    
#MOTHERTONGUE_MOTHERTONGUE_CAT
catvar = "MOTHERTONGUE_MOTHERTONGUE_CAT"
vars = []
for name in header:
    if name.count('MOTHERTONGUE')>0 and name[-3:]!='CAT': vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)
    
#MARRIAGE_CAT
catvar = "MARITAL_CAT"
vars = ['SES_MARITAL_UNKNOWN','SES_UNMARRIED','SES_MARRIED','SES_SEPARATED','SES_DIVORCED','SES_WIDOW','SES_REGPARTNERSHIP',
        'SES_DIVORCED_REGPARTNERSHIP','SES_WIDOW_REGPARTNERSHIP']#['SES_SELFEMPLOYED','SES_UPPERLEVEL','SES_LOWERLEVEL','SES_MANUAL','SES_STUDENTS','SES_PENSIONERS','SES_OTHERS','SES_NA']
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)
    
#SES_OCCUPATION_CAT
catvar = "OCCUPATION_CAT"
vars = []
for name in header:
    if name.count('OCCUPATION')>0 and name[-3:]!='CAT': vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
print(vars)
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)
    
#EDUCATION_CAT
catvar = "EDUCATION_CAT"
vars = []
for name in header:
    if name[:3]=='EDU' and name[-3:]!='CAT': vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)

#SES_EDUFIELD_CAT
#catvar = "EDUCATION_CAT"
#vars = []
#for name in header:
#    if name.count('EDUFIELD')>0 and name[-3:]!='CAT': vars.append(name)
        
#with open(outdir+catvar+'_variables.csv','wt') as outfile:
#    w = csv.writer(outfile,delimiter=',')
#    w.writerow(list(always_include)+vars)
    
#DRUG_CAT
catvar = "DRUG_DRUG_CAT"
vars = []
for name in header:
    if name[:5].count('DRUG_')>0 and name not in omit_set: vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)
    
#DISEASE_CAT
catvar = "DISEASE_DISEASE_CAT"
vars = []
for name in header:
    if (name.count('DISEASE_')>0 or name.count('INF')>0) and name not in omit_set: vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)
cat_variables[catvar] = vars    
#EARNINGS_CAT
catvar = "EARNINGS_CAT"
vars = ['EARNINGS_TOT','EARNINGS_NA']
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)
    

#SOCIALASSISTANCE_CAT
catvar = "SOCIALASSISTANCE_CAT"
vars = []
for name in header:
    if name[:16]=='SOCIALASSISTANCE': vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)

#SOCIALWELFARE_CAT
catvar = "SOCIALWELFARE_CAT"
vars = []
for name in header:
    if name[:13]=='SOCIALWELFARE': vars.append(name)
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)

#BIRTH_CAT
catvar = "BIRTH_CAT"
vars = []
for name in header:
    if name[:5]=='BIRTH': vars.append(name)
vars = list(set(vars))
all_vars += vars
cat_variables[catvar] = vars        
with open(outdir+catvar+'_variables.csv','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(always_include)+vars)    

end = time()
print("Input variable lists for categorical variables written in "+str(end-start)+" s")
```


```python
import itertools

start = time()
count = 0
outdir = '/data/projects/vaccination_project/data/permute_var_names_092022/'
removedir = '/data/projects/vaccination_project/data/remove_var_names_092022/'
#create files containing all possible combinations of variable categories to permute
categories = list(cat_variables.keys())
comb_varnames = []
#all_vars = set(["SEX","age_october_2021"])
for l in range(1,len(categories)+1):
    for subset in itertools.combinations(categories, l):
        s = ""
        for sub in subset: s += sub+"_"
        comb_varnames.append(s)
        with open(outdir+s+"perm-names-"+str(l)+".csv",'wt') as outfile:
            w = csv.writer(outfile,delimiter=',')
            vars = []
            for sub in subset: vars += cat_variables[sub]
            #if len(all_vars.intersection(vars))>0: print(all_vars.intersection(vars))
            #all_vars.update(vars)
            w.writerow(vars)
        #print(outdir+s+"permute_names.csv written!")
        count += 1
end = time()
print(str(count)+" files written in "+str(end-start)+" s")

#save combined category names to a file
with open(outdir+"combined_category_names.txt",'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    for n in comb_varnames: w.writerow([n])
```


```python
print("Total number of variables in all categories = "+str(len(all_vars)))
tot_N_in_categories = 0
all_vars_in_categories = []
for key in categories:
    tot_N_in_categories += len(cat_variables[key])
    all_vars_in_categories += cat_variables[key]
print("Total number of vars in the categories dict = "+str(tot_N_in_categories))
print(len(set(all_vars)))
print(len(set(all_vars_in_categories)))
```


```python
#save a file that contains the baseline variables, plus all variables from each of the categories
with open(outdir+'predictors_from_each_category_plus_baseline.txt','wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(list(all_vars))
```

