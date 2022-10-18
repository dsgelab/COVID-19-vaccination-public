from time import time
import pandas as pd
import csv

####################################################################
#READ IN THE CURRENT STUDY POPULATION AND REMOVE DEATHS DURING 2020#
####################################################################

#other exclusion criteria than deaths during 2020 and covid diagnoses have been applied to this file:
fname = "/data/projects/vaccination_project/data/vaccination_project_minimalphenotype_082022.csv"
study_ids = set(pd.read_csv(fname,usecols=['FINREGISTRYID'])['FINREGISTRYID'].unique())
print("Number of initial study IDs: "+str(len(study_ids)))

#remove people who died during 2020
#read in IDs of people who have died before the end of year 2020
death_name = "/data/processed_data/sf_death/thl2021_2196_ksyy_tutkimus.csv.finreg_IDsp"
exclude_IDs = set()
with open(death_name,'rt') as infile:
    r = csv.reader(infile,delimiter=',')
    for row in r:
        ID = row[0]
        if ID=='TNRO': continue
        else: exclude_IDs.add(ID)

study_ids = study_ids-exclude_IDs
print("Number of IDs after removing deaths during 2020: "+str(len(study_ids))) #3255578 IDs

#########################################
#PREPROCESS INFECTIOUS DISEASES REGISTER#
#########################################

fname = "/data/processed_data/thl_infectious_diseases/infectious_diseases_2022-01-19.feather"
infectious_outname = "/data/projects/vaccination_project/data/vaccination_project_infectious_diseases_082022.csv"
covid_positive_outname = "/data/projects/vaccination_project/data/vaccination_project_infectious_diseases_042022_COVID+_only.csv"
idfile = "/data/projects/vaccination_project/data/vaccination_project_study_ids_082022.csv"

#read in the columns of interest
#ID = TNRO = column 1
#recording_week = column 25
#reporting_group = column 26
#sampling_date
#NOTE: We define COVID positives as follows:
#reporting_group = "['Koronavirus', '--COVID-19-koronavirusinfektio']"
#reporting_week is between 1/2020 and 43/2021 (inclusive)
#if reporting week is NaN, then use sampling_date
#if also sampling date is NaN, mark as Covid positive (5 such cases)

df = pd.read_feather(fname,columns=['TNRO','recording_week','reporting_group','sampling_date'])
df['reporting_group'] = df['reporting_group'].astype(str)
df['recording_week'] = df['recording_week'].astype(str)
df['sampling_date'] = df['sampling_date'].astype(str)
print(df.head())

#first define COVID positive cases according to the definition above
include_col = []
for index,row in df.iterrows():
	
	if row['recording_week'].count('None')>0:
		#get the corresponding sampling date
		date = row['sampling_date']
		if date.count('None')>0: include_col.append(1) #if also sampling date is missing, assume this happened during time interval of interest as it is the more likely option
		else:
			#check if sampling date is within the time interval of Jan/2020-Oct/2021
			year = int(date.split('-')[0])
			month = int(date.split('-')[1])
			if year==2020 or (year==2021 and month<11): include_col.append(1)
			else: include_col.append(0) 
		continue
	year = int(row['recording_week'].split('/')[1])
	week = int(row['recording_week'].split('/')[0])
	if year==2020: include_col.append(1)
	elif week<=43: include_col.append(1)
	else: include_col.append(0)

df_COVID = df.copy()
df_COVID['include'] = include_col
print(df_COVID.head())

#subset to those cases that happened before 44/2021 and after the end of 2019
df_COVID = df_COVID.loc[df_COVID['include']>0]
print(df_COVID.head())

#get all unique values of the 'reporting_group' column
uniq_reporting_group = df_COVID['reporting_group'].unique()

#get the reporting group for COVID
for g in uniq_reporting_group:
	if g.count('COVID')>0:
		COVID_group = g
		break

#select only the rows that correspond to COVID_group
df_COVID = df_COVID.loc[df_COVID['reporting_group']==COVID_group]
#This df now contains everyone with a COVID diagnosis that we want to exclude from the study
#saving this list to a file
df_COVID.to_csv(covid_positive_outname,index=False)

#remove from study ids those that have a reported positive covid test
study_ids = list(study_ids - set(df_COVID['TNRO'].unique()))
print('Number of study IDs after removing COVID positives: '+str(len(study_ids))) #this is 3195308 IDs
study_ids.sort()
#save the final list of study ids to a file
with open(idfile,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    w.writerow(study_ids)

#Then create the variables used as predictors. We select the 10 most prevalent infectious diseases
#not counting COVID

#make variables of the top 10 most frequent reporting groups
#also map these levels to more interpretable names

reporting_group_map = {"['Klamydia']":"INF_CHLAMYDIA","['Influenssa' '--Influenssa A' '----Ei H1N1 eikä H5N5']":"INF_INFLUENZA_A","['Campylobacter']":"INF_CAMPYLOBACTER",
"['--C. difficile TOKS' 'C. difficile']":"INF_CLOSTRIDIOIDES_DIFFICILE","['Salmonella' '--Salmonella muu']":"INF_SALMONELLA",
"['RSV']":"INF_RSV","['ESBL-kantajuus' '--ESBL-kantajuus E.coli']":"INF_ESBL_CARRIER","['Influenssa' '--Influenssa B']":"INF_INFLUENZA_B","['M. pneumoniae']":"INF_MYCOPLASMA_PNEUMONIA",
"['Norovirus' 'Pieni pyöreä virus']":"INF_NOROVIRUS","['Hepatiitti C']":"INF_HEPATITIS_C","['Puumalavirus']":"INF_PUUMALAVIRUS",
"['Bakteerit' '--Grampositiiviset bakteerit' '----Stafylokokit'\n '------Staphylococcus aureus'\n '--------Staphylococcus aureus muu kuin MRSA' 'S. aureus, veri/likvor'\n '--S. aureus, veri/likvor ei MRSA']":"INF_STAPHYLOCOCCUS_AUREUS_TYPICAL",
"['MRSA-kantajuus']":"INF_MRSA_CARRIER"}

#All of these are used as separate variables.
#remove all diagnoses that happen after week 43/2021
#remove also IDs that are not in the study population
study_ids = set(study_ids)

include_col = []
for index,row in df.iterrows():
        ID = row['TNRO']
        if ID not in study_ids:
            #remove IDs not in the study population
            include_col.append(0)
            continue
        if row['recording_week'].count('None')>0:
                #get the corresponding sampling date
                date = row['sampling_date']
                if date.count('None')>0: include_col.append(1) #if also sampling date is missing, assume this happened during time interval of interest as it is the more likely option
                else:
                        #check if sampling date is before or after Oct/2021
                        year = int(date.split('-')[0])
                        month = int(date.split('-')[1])
                        if year<2021 or (year==2021 and month<11): include_col.append(1)
                        else: include_col.append(0)
                continue
        year = int(row['recording_week'].split('/')[1])
        week = int(row['recording_week'].split('/')[0])
        if year<2021: include_col.append(1)
        elif week<=43: include_col.append(1)
        else: include_col.append(0)

df['include'] = include_col

#subset to those cases that happened before 44/2021
df = df.loc[df['include']>0]
print(df.head())

#save the infectious diseases data
#Create the individual variables
inf_IDs = set(df['TNRO']) #IDs found from the infectious diseases register
missing_ids = set(study_ids)-inf_IDs
print('Number of study population IDs missing from infectious diseases register: '+str(len(missing_ids)))
varnames = [reporting_group_map[key] for key in reporting_group_map]
rowcount = 0
data = {} #key = ID, value = {FINREGISTRYID:ID,var1:val1,var2:val2,...}

#first add the ids with entries in the infectious diseases register
for index,row in df.iterrows():
    ID = row['TNRO']
    rowcount += 1
    if rowcount%100000<1: print(str(rowcount)+" rows processed")
    #if ID not in study_ids: continue
    if ID not in data:
        data[ID] = {'FINREGISTRYID':ID}
        for g in reporting_group_map: data[ID][g] = 0
    reporting_group = row['reporting_group']
    #only use the most frequent reporting groups defined above as variables#
    if reporting_group not in reporting_group_map: continue
    data[ID][reporting_group] = 1

#then add the missing ids as empty rows
for ID in missing_ids:
    data[ID] = {'FINREGISTRYID':ID}
    for g in reporting_group_map: data[ID][g] = 0 
        
#convert to dataframe
data_df = pd.DataFrame.from_dict(data,orient='index')
data_df = data_df.rename(columns=reporting_group_map)
#save the resulting dataframe to a file
data_df.to_csv(infectious_outname,index=False)    
    
##############################
#PREPROCESS MARRIAGE REGISTER#
##############################

fname = "/data/processed_data/dvv/Tulokset_1900-2010_tutkhenk_aviohist.txt.finreg_IDsp"
marriage_outname = "/data/projects/vaccination_project/data/vaccination_project_marriage_082022.csv"

#read in the columns of interest
#FINREGISTRYID = column 0
#Current_marital_status = column 1
#Starting_date = column 5

#For each ID, we take the newest current_marital_status
#Newest is the one where Starting_date is the latest
#Map the marital status variables into more easily interpretable names:
#Columns of the output file are:
#FINREGISTRYID
#SES_MARITALSTATUS_CAT = Marital status as a categorical variable, see levels from below
#SES_MARITAL_UNKNOWN = Marital status unknown (Current_marital_status=0)
#SES_UNMARRIED = unmarried (Current_marital_status=1)
#SES_MARRIED = married (Current_marital_status=2)
#SES_SEPARATED = separated (Current_marital_status=3)
#SES_DIVORCED = divorced (Current_marital_status=4)
#SES_WIDOW = widowed (Current_marital_status=5)
#SES_REGPARTNERSHIP = registered partnership (Current_marital_status=6)
#SES_DIVORCED_REGPARTNERSHIP = divorced from registered partnership (Current_marital_status=7)
#SES_WIDOW_REGPARTHERSHIP = widowed from registered partnership (Current_marital_status=8)


df_marriage = pd.read_csv(fname,usecols=['FINREGISTRYID','Current_marital_status','Starting_date'])

from datetime import datetime
eofu = datetime.strptime('Oct 31 2021', '%b %d %Y')

#we want to keep only the latest entry that started before the end of October 2021
df_marriage['Starting_date'] = pd.to_datetime(df_marriage['Starting_date'])
df_marriage = df_marriage.loc[df_marriage['Starting_date']<eofu]
df_marriage = df_marriage.sort_values('Starting_date').groupby('FINREGISTRYID').tail(1)

import csv
#save the resulting file
marriage_ids = set(df_marriage['FINREGISTRYID'].unique())
missing_ids = set(study_ids)-marriage_ids
print('Number of study population IDs missing from marriage register: '+str(len(missing_ids)))
with open(marriage_outname,'wt') as outfile:
    w = csv.writer(outfile,delimiter=',')
    #write header
    header = ['FINREGISTRYID','SES_MARITALSTATUS_CAT','SES_MARITAL_UNKNOWN','SES_UNMARRIED','SES_MARRIED','SES_SEPARATED','SES_DIVORCED','SES_WIDOW','SES_REGPARTNERSHIP',
              'SES_DIVORCED_REGPARTNERSHIP','SES_WIDOW_REGPARTNERSHIP']
    w.writerow(header)
    for ID in missing_ids: w.writerow([ID,0,1,0,0,0,0,0,0,0,0])
    print('missing IDs written')
    for index,row in df_marriage.iterrows():
        ID = row['FINREGISTRYID']
        if ID not in study_ids: continue
        outrow = [0 for i in range(len(header))]
        val = row['Current_marital_status']
        outrow[0] = ID
        outrow[1] = val
        outrow[val+2] = 1
        w.writerow(outrow)
