library(data.table)
library(dplyr)
library(tidyr)
library(stringr)

#########################################################################
## PROCESS WIDE ENDPOINTER FILES TO CREATE A yes/no ENDPOINT INDICATOR ##
#########################################################################


## Read in the endpointer file
first_row <- fread(cmd = paste0('head -n 1 ', '/data/processed_data/endpointer/wide_first_events_endpoints_2021-12-20_no_OMITS.txt'))
to_select <- colnames(first_row)[grepl("_NEVT$",colnames(first_row)) & !grepl("EXALLC|EXMORE",colnames(first_row))]

e <-fread('/data/processed_data/endpointer/wide_first_events_endpoints_2021-12-20_no_OMITS.txt', select=c("FINREGISTRYID",to_select))

# Write temporary file to free up memory
fwrite(e, file="/home/aganna/temp.txt", sep=" ")

# Process in awk
# awk '{if (NR>1) {for (C=2; C<=NF; C++) {if ($C > 1) {$C=1}}} print}' /home/aganna/temp.txt > /home/aganna/temp2.txt

# Now read int
e <-fread('/home/aganna/temp2.txt')

# Remove columns if the total number of individuals with he endpoint < 1000,F5_SAD_NEVT is removed because not numeric 

to_select <- colnames(e)[!colnames(e) %in% c("F5_SAD_NEVT","FINREGISTRYID")]
et <- e[,..to_select]
count_columns <- et[, lapply(.SD,sum)]
es <- e %>% select(FINREGISTRYID,names(count_columns)[as.numeric(count_columns)>1000])


fwrite(es, file="/home/aganna/wide_first_events_endpoints_dicot.csv")



#########################################################
## PROCESS DRUG FILE TO CREATE A yes/no DRUG INDICATOR ##
#########################################################

# Do this in unix
# awk 'BEGIN {FS=","} NR==1; NR>1{if($2 == "PURCH") print $1,$3,$4,$6}' /data/processed_data/detailed_longitudinal/detailed_longitudinal.csv | > /home/aganna/detailed_longitudinal_purchases_wide.txt

## Read in all the drugs and create and wide file
l <- fread("/home/aganna/detailed_longitudinal_purchases_wide.txt")
colnames(l) <- c("FINREGISTRYID","EVENT_AGE","PVM","CODE1")

# Number of unique drugs
length(unique(l$CODE1)) #1432

# Simplify to one drug for each person
lu <- unique(setDT(l), by = c("FINREGISTRYID", "CODE1"))
fwrite(lu, file="/home/aganna/detailed_longitudinal_purchases_wide_unique.csv")

lu <- fread("/home/aganna/detailed_longitudinal_purchases_wide_unique.csv")

# Exclude drugs seens in less than 1,000 individuals and with missing ATC code
drug_to_keep <- lu %>% count(CODE1) %>% filter(n>1000 & CODE1!="")
luf <- lu %>% filter(CODE1 %in% drug_to_keep$CODE1) %>% mutate(t=1)

# Reshape the file to get a wide format

# Split the problem in smaller chunks
uid <- unique(luf$FINREGISTRYID)
FINREGIDGROUP <- split(uid, ceiling(seq_along(uid)/100000))

for (k in 1:length(FINREGIDGROUP))
{
  to_keep <- FINREGIDGROUP[k]
  luft <- luf %>% filter(FINREGISTRYID %in% to_keep[[1]])
  luftF <- dcast(setDT(luft), FINREGISTRYID~CODE1, value.var=c('t'), fill=0)
  print(dim(luftF))
  fwrite(luftF, file=paste0("/home/aganna/temp/drug_purchases_binary_wide_",k,".csv"))
}

## Check that all the files have same columns
RES <- NULL
for (i in 1:length(FINREGIDGROUP))
{
  headd <- fread(cmd = paste0('head -n 1 ', '/home/aganna/temp/drug_purchases_binary_wide_',i,'.csv'))
  RES <- rbind(RES,colnames(headd))
  
}

apply(RES, 2, function(x) {length(unique(x)) == 1})

# Now cat the files (do this unix)

#head -1 /home/aganna/temp/drug_purchases_binary_wide_1.csv > /home/aganna/drug_purchases_binary_wide_ALL.csv
#tail -n +2 -q /home/aganna/temp/drug_purchases_binary_wide_*.csv >>c


########################################
## PROCESS THE MINIMAL PHENOTYPE FILE ##
########################################

#1.4.2022 update to use the new minimal phenotype file
#23.8.2022 update to remove people older than 80 already here

#Also exclude everyone with date of death prior to 2020-01-01 (col 4)
#columns to include in the final file:
#1 - FINREGISTRYID
#2 - date_of_birth
#3 - sex
#4 - post_code_last
#5 - mother_tongue
#6 - ever_married
#7 - ever_divorced
#8 - emigrated
#9 - in_social_hilmo
#10 - in_social_assistance_registries
#11 - number_of_children
#12 - drug_purchases

select_columns <- c("FINREGISTRYID","index_person","date_of_birth","death_date","sex","post_code_last","mother_tongue","ever_married","ever_divorced","emigrated","in_social_hilmo","in_social_assistance_registries","number_of_children","drug_purchases")
mf <- fread("/data/processed_data/minimal_phenotype/minimal_phenotype_2022-03-28.csv",select=select_columns) #7166416 rows
#remove people who died before 2020-01-01
mf <- mf[is.na(mf$death_date),] #5574194 rows
mf <- subset(mf,date_of_birth<as.Date('1991-01-11')) #we only analyze people older than 30, 3852666 rows
mf <- subset(mf,date_of_birth>as.Date('1941-01-11')) #we only analyze people younger than 80, 3492506 rows
mf <- subset(mf,emigrated<1) #Only use people who have not emigrated, 3278800 rows
#remove index_person and death_date columns
mf <- mf[,!c("index_person","death_date")]

fwrite(mf,file="/data/projects/vaccination_project/data/vaccination_project_minimalphenotype_082022.csv")

#############################################################
## PROCESS VACCINATION REGISTER TO CREATE OUTCOME VARIABLE ##
#############################################################

v <- fread("/data/processed_data/thl_vaccination/vaccination_2022-05-10.csv")

#Definition - Part 1: Anything containing "cov", "cor", "kor" "mod", "astra", "co19", "cvid", "cominarty" - Remove cases of dukoral, ticovac, vesirokkorokote
##Make all entries lower case letters with no spaces for consistency
v$LAAKEAINE_SELITE <- tolower(v$LAAKEAINE_SELITE)
v$LAAKEAINE_SELITE <- gsub(" ", "", v$LAAKEAINE_SELITE, fixed=TRUE)

v$COVDEF <- case_when(grepl("cov|cor|kor|mod|astra|co19|cvid|com", v$LAAKEAINE_SELITE) & !(grepl("dukoral|ticovac|vesiro", v$LAAKEAINE_SELITE)) ~ 1,
                      TRUE ~ 0)

#Definition - Part 2: DRUG == J07BX03
v$COVDEF[v$LAAKEAINE == "J07BX03"] <- 1

#Definition - Part 3: VACCINE_PROTECTION_1 == 29 
v$COVDEF[grepl("29",v$ROKOTUSSUOJA)] <- 1

#Subset to covid vaccines only
covidVax <- subset(v, COVDEF==1)

#First covid vaccinations began in Finland on 27th December 2020 -
covidVax <- subset(covidVax, KAYNTI_ALKOI >= "2020-12-27")

#Identify number of doses per individual
#People with only one vaccine should have no difference in dates 
covidDose <- covidVax %>%
  group_by(TNRO) %>%
  summarise(first_visit = min(KAYNTI_ALKOI), last_visit = max(KAYNTI_ALKOI))

covidDose$difference <- covidDose$last_visit - covidDose$first_visit  

#Get difference in number between 6 weeks (42 days) and 8 weeks (56 days)
covidDose$doseno56 <- case_when(covidDose$difference >= 56 ~ 2,TRUE ~ 1)
covidDose$doseno42 <- case_when(covidDose$difference >= 42 ~ 2,TRUE ~ 1) 

#Subset to people aged 30 (Run sensitivity analysis where you subset to people aged 40?)
#here, get the IDs from the updated minimal phenotype file - Tuomo 1.4.2022
mf_name <- "/data/processed_data/minimal_phenotype/minimal_phenotype_2022-03-28.csv"
age <- fread(input = mf_name, select= c("FINREGISTRYID","date_of_birth","death_date","index_person"))


#Join number of covid doses to age (at 31 october 2021)
doses <- left_join(age, covidDose, by=c("FINREGISTRYID"="TNRO")) %>% mutate(age_october_2021=as.numeric(as.Date("2021-10-31")-as.Date(date_of_birth))/365.25)
doses$doseno56[is.na(doses$doseno56)] <- 0
doses$doseno42[is.na(doses$doseno42)] <- 0

#Age over 30
doses30 <- subset(doses, age_october_2021 >= 30 & is.na(death_date) & index_person==1) 
table(doses30$doseno56) 
table(doses30$doseno42) 
hist(doses30$difference[doses30$doseno56==1])

#Age over 40
doses40 <- subset(doses, age_october_2021 >= 40 & is.na(death_date) & index_person==1) 
table(doses40$doseno56) 
table(doses40$doseno42) 
hist(doses40$difference[doses40$doseno56==1])

#Final phenotype 
pheno <- doses30 %>% select(-death_date,-index_person,-doseno42)

#Coallesce 1 and 2 dose as only interested in 1+ vs 0 doses.
pheno$COVIDVax <- case_when(pheno$doseno56==1 | pheno$doseno56==2 ~ 0,
                            pheno$doseno56==0 ~ 1,
                            TRUE ~ NA_real_)

fwrite(pheno,file="/data/projects/vaccination_project/data/vaccination_outcome_including_covid19+_052022.csv")


#Read in COVID cases and exclude - chosen the phenotype -
#covidcase <- fread("zcat < /finngen/red/Bradley/COVID_GWAS/COVIDPhenoAndCov.txt.gz", data.table=FALSE)
#covidcaseB1 <- subset(covidcase, B1COVID == 1 | B1COVID == 0) 
#covidcaseB1 <- covidcaseB1[,c("FINNGENID")]

#covidcaseC2 <- subset(covidcase, C2COVID == 1) 
#covidcaseC2 <- covidcaseC2[,c("FINNGENID")]
