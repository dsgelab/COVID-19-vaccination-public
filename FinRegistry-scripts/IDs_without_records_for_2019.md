```python
#get all IDs with any records for year 2019 in the following data
Drug_purchases_file = "/data/processed_data/kela_purchase/175_522_2020_LAAKEOSTOT_2019.csv.finreg_IDsp"
Disease_diagnoses_file = "/data/processed_data/endpointer/longitudinal_endpoints_2021_12_20_no_OMITS.txt"
Income_file = "/data/processed_data/etk_pension/vuansiot_2022-05-12.csv"
Social_benefits_file = "/data/processed_data/thl_social_assistance/3214_FinRegistry_toitu_MattssonHannele07122020.csv.finreg_IDsp"
Long_term_care_file = "/data/processed_data/thl_soshilmo/thl2019_1776_soshilmo.csv.finreg_IDsp"
Birth_registry_file = "/data/processed_data/thl_birth/birth_2022-03-08.csv"

#first drug purchases
import csv

ids_2019 = set() #set of IDs with any record during 2019
with open(Drug_purchases_file,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        if row[0].count('HETU')>0: continue
        frid = row[0]
        year = int(row[2].split('-')[0])
        if year==2019: ids_2019.add(frid)
print("Number of IDs with drug purchases in 2019 is "+str(len(ids_2019)))
```


```python
#Then disease diagnoses
with open(Disease_diagnoses_file,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        if row[0].count('FINREGISTRYID')>0: continue
        frid = row[0]
        year = int(row[3])
        if year==2019: ids_2019.add(frid)
print("Number of IDs with drug purchases in 2019 is "+str(len(ids_2019)))
```


```python
#income
with open(Income_file,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        if row[0].count('laki')>0: continue
        frid = row[5]
        year = int(row[1])
        vuosiansio_indexed = float(row[6])
        if year==2019 and vuosiansio_indexed>0: ids_2019.add(frid)
print("Number of IDs with a data entry in 2019 is "+str(len(ids_2019)))
```


```python
#social benefits
with open(Social_benefits_file,'rt') as infile:
    r = csv.reader(infile,delimiter=';')
    for row in r:
        if row[0].count('TNRO')>0: continue
        frid = row[0]
        year = int(row[1])
        if year==2019: ids_2019.add(frid)
print("Number of IDs with a data entry in 2019 is "+str(len(ids_2019)))
```


```python
#lon-term care
with open(Long_term_care_file,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        if row[0].count('TNRO')>0: continue
        frid = row[0]
        year = int(row[1])
        if year==2019: ids_2019.add(frid)
print("Number of IDs with a data entry in 2019 is "+str(len(ids_2019)))
```


```python
#birth registry
with open(Birth_registry_file,'rt') as infile:
    r = csv.reader(infile,delimiter=';')
    for row in r:
        if row[0].count('AITI_TNRO')>0: continue
        frid = row[0]
        year = int(row[3])
        if year==2019: ids_2019.add(frid)
print("Number of IDs with a data entry in 2019 is "+str(len(ids_2019)))
```


```python
#then read in all COVID vaccination study IDs
study_ids = set()
with open('/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_no_Askola.csv.newheader','rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        if row[0].count('FINREGISTRYID')>0: continue
        study_ids.add(row[0])
print("Number of study IDs is "+str(len(study_ids)))
```


```python
#get the list of IDs in the vaccination study population that have no data entries for 2019
no_data_2019_ids = study_ids-ids_2019
print('Number of vaccination study IDs without data in 2019 is '+str(len(no_data_2019_ids))+"/"+str(len(study_ids)))
```


```python
#save these IDs to a file
with open("/data/projects/vaccination_project/data/vaccination_study_ids_with_no_data_in_2019.csv",'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    for frid in no_data_2019_ids: w.writerow([frid])
```


```python
from time import time
start = time()
#filter the IDs in no_data_2019_ids from the test set file

test_file = "/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_test.csv.newheader"
test_file_filtered = "/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_test_nodata2019removed.csv.newheader"

with open(test_file,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    with open(test_file_filtered,'wt') as outfile:
        w = csv.writer(outfile,delimiter=',')
        for row in r:
            if row[0] not in no_data_2019_ids: w.writerow(row)
                
end = time()
print('Filtering done in '+str(end-start)+" s")
```


```python
start = time()
#filter the IDs in no_data_2019_ids from the training set file

train_file = "/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_train.csv.newheader"
train_file_filtered = "/data/projects/vaccination_project/data/vaccination_project_combined_variables_wide_30082022_train_nodata2019removed.csv.newheader"

with open(train_file,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    with open(train_file_filtered,'wt') as outfile:
        w = csv.writer(outfile,delimiter=',')
        for row in r:
            if row[0] not in no_data_2019_ids: w.writerow(row)
                
end = time()
print('Filtering done in '+str(end-start)+" s")
```

