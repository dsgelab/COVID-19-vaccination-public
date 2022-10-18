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
	make_option(c("-n", "--nochunk"), type="character", default="FALSE",
              help="If TRUE, pass all training data to bigglm as one chunk [default=FALSE].", metavar="character"),
	make_option(c("-N", "--Nsamples"), type="integer", default=10,
              help="Number of independent samples drawn for estimating AUC on the test data [default=10].", metavar="integer"),
	make_option(c("-c", "--isCat"), type="character", default="no",
              help="If yes, one model is fit with all variables defined by selectfile [default=no].", metavar="character"),
  make_option(c("-C", "--catName"), type="character", default="CAT",
              help="Name of the categorical variable (only used if isCat = yes).",metavar="character"),
	make_option(c("-l", "--shufflefile"), type="character", default=NULL,
	            help="Name of the file containing the columns to be shuffled.",metavar="character"),
	make_option(c("-f", "--fraction"), type="double", default=0.75,
              help="Fraction of the test set rows used in each independent sample when estimating AUC [default=0.75].", metavar="double")
	
); 
 
#parse input arguments, no checks done
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

#load needed libraries
library(dplyr)
library(data.table)
library(caret)
library(pROC)
library(glmnet)
library(Matrix)

options(device='Cairo')

start_time <- Sys.time()

# create a function to transform coefficient of glmnet and cvglmnet to data.frame
coeff2dt <- function(fitobject, s) {
  coeffs <- coef(fitobject, s)
  coeffs.dt <- data.frame(name = coeffs@Dimnames[[1]][coeffs@i + 1], coefficient = coeffs@x)

  # reorder the variables in term of coefficients
  return(coeffs.dt[order(coeffs.dt$coefficient, decreasing = T),])
}

#read in data
select_columns <- scan(opt$selectfile,character(),quote="",sep=",") 

print('Data read in.')

#Skip the following variables as these are used to adjust the models
skips <- c("SEX","age_october_2021","COVIDVax")
#Iterate over each variable and fit a logistic regression adjusted for age and sex.
models <- c()
aucs <- c()
stds <- c()
'%nin%' = Negate('%in%')


if (opt$isCat=='yes')
{
        variable <- opt$catName
        print('Starting to analyze using one model for the whole input data')
	#training data is read in in chunks to save memory
	for (i in 0:9){
		print(i)
		train_data1 <- fread(file=paste0('/data/projects/vaccination_project/data/ml_downsampled_data_30082022/vaccination_project_combined_variables_wide_30082022_train_downsampled_imputed_split',i,'_nodata2019removed.csv.newheader'),select=select_columns,data.table=FALSE)

    #shuffle values in columns provided with the shufflefile command line flag
    if (!is.null(opt$shufflefile)){
      #read in the column names to be shuffled
      shuffle_columns <- scan(opt$shufflefile,character(),quote="",sep=",")
      for (colname in shuffle_columns){
        train_data1[colname] <- sample(train_data1[,colname])  
      }
    }
        print(paste0('Number of ones in split=',sum(train_data1$COVIDVax),', total size of split=',nrow(train_data1)))
        
		#reformat to sparse matrices and combine with other input data slices
		trainX1 <- as(data.matrix(subset(train_data1,select=-COVIDVax)),"sparseMatrix")
		trainY1 <- as(data.matrix(train_data1$COVIDVax),"sparseMatrix")
		rm(train_data1)
		if (i==0){
			trainX <- trainX1
			trainY <- trainY1
		}
		if (i>0){
			trainX <- rbind(trainX,trainX1)
			trainY <- rbind(trainY,trainY1)
		}
		rm(trainX1)
		rm(trainY1)
	}


    #compute class weights
    nnz <- diff(trainY@p) #number of non-zeros
    nz <- nrow(trainY)-nnz #number of zeros
    
    fraction_0 <- nz/nrow(trainY)
    fraction_1 <- nnz/nrow(trainY)
    # assign that value to a "weights" vector
    weights <- numeric(nrow(trainY))
    weights[which(trainY!=1, arr.ind=TRUE)] <- fraction_0
    weights[which(trainY==1, arr.ind=TRUE)] <- fraction_1

	#fit the glmnet model
  cv.fit <- cv.glmnet(trainX, trainY, parallel=FALSE, weights=weights)
  print('model fit')

  rm(trainX)
  rm(trainY)
  #save the coefficients of the best glmnet model to a file
  coeffs <- coeff2dt(fitobject = cv.fit, s = "lambda.min")
  write.csv(coeffs,paste0(opt$outprefix,variable,"_best_model_coeffs.csv"))
  print('Model coefficients saved.')

  #and save the model to a file
  outname <- paste0(opt$outprefix,variable,"_glmnet_model.RData")
  save(cv.fit,file=outname)
  print('Model saved to a file.')

	test_data <- fread(file=opt$testfile,select=select_columns,data.table=FALSE)
	#USING THE PRE-IMPUTED TEST DATA

	#subset to younger than 80 yo
	test_data <- subset(test_data,age_october_2021<80)

	#Then compute AUC on --Nsample independent samples from the test set
	testX <- as(data.matrix(subset(test_data,select=-COVIDVax)),"sparseMatrix")
        testY <- as(test_data$COVIDVax,"sparseMatrix")
        predicted <- predict(cv.fit,testX,type="response")
        #add the predicted values to test_data as a new column
        test_data$predicted <- predicted

       	#write the predictions to a file
        write.csv(test_data,gzfile(paste0(opt$outprefix,variable,"_test_set_predictions.csv.gz")))
        #Compute auc and confience intervals
        auc <- auc(c(test_data$COVIDVax),c(test_data$predicted))#COVIDVax ~ predicted, data=test_data)
        write.csv(auc,paste0(opt$outprefix,variable,"_auc.csv"))
        auc_CI <- ci.auc(auc)#COVIDVax ~ predicted, data=test_data)
        write.csv(auc_CI,paste0(opt$outprefix,variable,"_auc_CIs.csv"))
        
        rm(testX)
        rm(testY)
}


if (opt$isCat=='no')
{
	train_data <- fread(file=opt$trainfile,select=select_columns,data.table=FALSE)
	#note that we are assuming pre-imputed data here

	test_data <- fread(file=opt$testfile,select=select_columns,data.table=FALSE)
	print('Variables to skip:')
	print(skips)
	for (variable in colnames(train_data)){
		models <- c()
		aucs <- c()
		stds <- c()
		
        	print(c('Starting to analyze ',variable))
        	if (variable %nin% skips){
			#If variable only has zero values, skip it
			print(dim(train_data))
			print(dim(train_data[variable]))
			if (max(train_data[variable],na.rm=TRUE)<1) next
			
			#format the train and test data so that glmnet can eat them
			vars <- c('SEX','age_october_2021',variable)
			trainX <- data.matrix(subset(train_data,select=vars))
			trainY <- data.matrix(train_data$COVIDVax)
			testX <- data.matrix(subset(test_data,select=vars))
			testY <- test_data$COVIDVax

			#fit the glmnet model
			cv.fit <- cv.glmnet(trainX, trainY, parallel=FALSE)
                	print('model fit')

			rm(trainX)
			rm(trainY)
            
			#save the coefficients of the best glmnet model to a file
			coeffs <- coeff2dt(fitobject = cv.fit, s = "lambda.min")
			write.csv(coeffs,paste0(opt$outprefix,variable,"_best_model_coeffs.csv"))
			print('Model coefficients saved.')

			#and save the model to a file
			outname <- paste0(opt$outprefix,variable,"_glmnet_model.RData")
			save(cv.fit,file=outname)
			print('Model saved to a file.')			


			#Then compute AUC on --Nsample independent samples from the test set
			predicted <- predict(cv.fit,testX,type="response")
			#add the predicted values to test_data as a new column
			test_data$predicted <- predicted
			
			#write the predictions to a file
			write.csv(test_data,gzfile(paste0(opt$outprefix,variable,"_test_set_predictions.csv.gz")))
			#Compute auc and confience intervals
			auc <- auc(c(test_data$COVIDVax),c(test_data$predicted))#COVIDVax ~ predicted, data=test_data)
			write.csv(auc,paste0(opt$outprefix,variable,"_auc.csv"))
			auc_CI <- ci.auc(auc)#COVIDVax ~ predicted, data=test_data)
			write.csv(auc_CI,paste0(opt$outprefix,variable,"_auc_CIs.csv"))
						
			rm(testX)
			rm(testY)
        	}
	}
}

end_time <- Sys.time()
write.csv(end_time-start_time,paste0(opt$outprefix,variable,"_runtime.csv"))
print(paste0("Run time: ",end_time-start_time))
