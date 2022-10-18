#!/usr/bin/env /opt/R-4.1.1/bin/Rscript

library(optparse)

options(warn=1)

option_list = list(
  	make_option(c("-r", "--trainfile"), type="character", default=NULL, 
              help="Full path to the training data.", metavar="character"),
	make_option(c("-e", "--testfile"), type="character", default=NULL, 
              help="Full path to the test data.", metavar="character"),
	make_option(c("-s", "--selectfile"), type="character", default=NULL, 
              help="Full path to the file containing column names used as variables in the logistic regression models.", metavar="character"),
    	make_option(c("-o", "--outprefix"), type="character", default="./", 
              help="Output prefix [default= ./].", metavar="character"),
	make_option(c("-l", "--reflevelfile"), type="character", default=NULL,
              help="File containing variable names used as reference for variable categories. If no reference level variable is given in this file, all other values are used as controls for each variable.", metavar="character"),
	make_option(c("-c", "--isCat"), type="character", default="no",
              help="If yes, one model is fit with all variables defined by selectfile [default=no].", metavar="character"),
	make_option(c("-C", "--catName"), type="character", default="CAT",
              help="Name of the categorical variable (only used if isCat = yes).",metavar="character"),
	make_option(c("-n", "--nochunk"), type="character", default="FALSE",
              help="If TRUE, pass all training data to bigglm as one chunk [default=FALSE].", metavar="character"),
	make_option(c("-N", "--Nsamples"), type="integer", default=10,
              help="Number of independent samples drawn for estimating AUC on the test data [default=20].", metavar="integer"),
	make_option(c("-f", "--fraction"), type="double", default=0.75,
              help="Fraction of the test set rows used in each independent sample when estimating AUC [default=0.75].", metavar="double")
); 
 
#parse input arguments, no checks done
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

#load needed libraries
library(tidyr)
library(dplyr)
library(data.table)
library(caret)
library(pROC)
library(biglm)

#First read in the names of the variables that are used as control/reference levels
#These variables must always be read in
ref_variables <- fread(opt$reflevelfile,sep=",",data.table=FALSE,header=FALSE)
#first read in the endpoints that should be used according to Andrius' work, others are skipped
used_endpoint_names <- scan('/data/projects/vaccination_project/data/omitted_endpoints_from_Andrius.csv',character(),sep=",")

#read in data
select_columns <- scan(opt$selectfile,character(),quote="",sep=",") 
#add reference level variable names to select_columns if they are not in there already
select_columns <- unique(c(select_columns,ref_variables[,2]))

train_data <- fread(file=opt$trainfile,select=select_columns,data.table=FALSE)
#subset to younger than 80 yo
train_data <- subset(train_data,age_october_2021<80)
#USE PRE-IMPUTED DATA

print('Data read in.')

#Skip the following variables as these are used to adjust the models
skips <- c("SEX","age_october_2021","COVIDVax",ref_variables[,2])

#Iterate over each variable and fit a logistic regression adjusted for age and sex.
models <- c()
aucs <- c()
stds <- c()
'%nin%' = Negate('%in%')

if (opt$isCat=='yes')
{
	variable <- opt$catName
	print('Starting to analyze using one model for the whole input data')
	mod <- as.formula(paste("COVIDVax ~", paste(select_columns[!select_columns %in% "COVIDVax"], collapse = " + ")))
	print(mod)
  	if (opt$nochunk=='TRUE') m <- bigglm(mod , data=train_data)
  	if (opt$nochunk=='FALSE') m <- bigglm(mod , data=model.matrix(mod,train_data),chunksize=nrow(train_data))
  	print('model fit')
	#save model information into dataframe models
  	models <- rbind(models,summary(m)$mat)
  	#and save the model to a file
  	outname <- paste0(opt$outprefix,variable,"_042022_glm_model.RData")
  	save(m,file=outname)
	
}
if (opt$isCat=='no')
{
	print('Variables to skip:')
	print(skips)
	for (variable in colnames(train_data)){
		models <- c()
		aucs <- c()
		stds <- c()
		#test if this variable corresponds to an endpoint that should be omitted
		if (startsWith(variable,'DISEASE')){
			if(!(variable %in% used_endpoint_names)) {
				print(paste0('skipping ',variable))
				next
			}
		}
        	print(c('Starting to analyze ',variable))
        	if (variable %nin% skips){
			#If variable only has zero values, skip it
			print(dim(train_data))
			print(dim(train_data[variable]))
			if (max(train_data[variable],na.rm=TRUE)<1) next
			#If the variable category has a specified reference level, use only those controls that match
			#to this reference variable value
			rowind <- 1
			ref_changed <- FALSE
			for (prefix in ref_variables[,1]){
				if (startsWith(variable,prefix)){
					refvar <- ref_variables[rowind,2]
					#print(summary(train_data))
					print(c("Reference level = ",refvar))
					
					train_data_i <- subset(train_data,refvar>0 | variable>0)
					ref_changed <- TRUE
					break
				}
				rowind <- rowind+1
			}
			if (ref_changed==FALSE){
				train_data_i <- train_data
			}
            #only select the needed variables to train_data_i and omit NAs
            train_data_i <- train_data_i[c("COVIDVax","SEX","age_october_2021",variable)]
            print('NA count before filtering:')
            print(sum(is.na(train_data_i[,variable])))
            train_data_i <- na.omit(train_data_i)
            print('NA count after filtering:')
            print(sum(is.na(train_data_i[,variable])))
            print("train_data_i size:")
            print(dim(train_data_i))
            zeros <- colSums(train_data_i[variable]<1)
            print(paste0('Number of zeros: ',zeros))
            print('Unique values for sex:')
            print(unique(train_data_i['SEX']))
            #For birth registry variables we only have women, so sex cannot be used for adjusting
            if(grepl('BIRTH_',variable,fixed=TRUE)){
                mod <- as.formula(sprintf("COVIDVax ~ age_october_2021 + %s", variable))
            }
            else{
                mod <- as.formula(sprintf("COVIDVax ~ SEX + age_october_2021 + %s", variable))
            }
		#if (opt$nochunk=='FALSE') mod <- as.formula(sprintf("COVIDVax ~ SEX + age_october_2021 + %s", variable),chunksize=nrow(train_data_i))
      		print(mod)
            print(nrow(train_data_i))
            #m <- glm(mod , data=train_data_i,family='binomial')
            m <- bigglm(mod , data=train_data_i)
            print(m)
            print(summary(m))
            #m <- bigglm(mod , data=train_data_i,chunksize=nrow(train_data_i))
		#print(train_data_i)
      #		if (opt$nochunk=='FALSE') m <- bigglm(mod , data=train_data_i)
        #if (opt$nochunk=='TRUE') m <- bigglm(mod , data=train_data_i,chunksize=nrow(train_data_i))
      		print('model fit')
      		#save model information into dataframe models
      		models <- rbind(models,summary(m)$mat)
            print(models)
		#save the results into a file
		write.csv(models,paste0(opt$outprefix,variable,"_coefficients.csv"))
      		#and save the model to a file
      		outname <- paste0(opt$outprefix,variable,"_042022_glm_model.RData")
      		save(m,file=outname)
		print('model saved')

        	}
	}
}
