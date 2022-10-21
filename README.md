# Predictors of COVID-19 vaccination status

This repository contains the essential analysis code used to produce the results in the manuscript:

Tuomo Hartonen, Bradley Jermy, Hanna Sõnajalg, Pekka Vartiainen, Kristi Krebs, Andrius Vabalas, FinnGen, Estonian Biobank Research Team, Tuija Leino, Hanna Nohynek, Jonas Sivelä, Reedik Mägi, Mark Daly, Hanna M. Ollila, Lili Milani, Markus Perola, Samuli Ripatti, Andrea Ganna, "Health, socioeconomic and genetic predictors of COVID-19 vaccination uptake: a nationwide machine-learning study"

## Note on data access

Please note that the raw data is only accessible via the secure computing environments of FinRegistry, FinnGen and Estonian biobank pending separate data access approvals, as described in the manuscript. More information about the data and data access can be found from:

* FinRegistry: https://www.finregistry.fi/
* FinnGen: https://www.finngen.fi/en
* Estonian biobank: https://genomics.ut.ee/en/content/estonian-biobank

## Code

The analysis code is divided into folders based on the datasets analysed. Code for performing the essential steps of the analyses is included.

### FinRegistry - code for the registry-based analyses

Some analysis scripts were run as Jupyter notebooks in the secure computing environment of the FinRegistry project. Due to security reasons, Jupyter notebooks cannot be exported from the computing environment, so these notebooks were converted to markdown and all output was removed before exporting. Thus some of the analysis code is included as markdown files.

#### Preprocessing the registry data

There are three main scripts used to preprocess the registry-based data. Preprocessing includes selecting and defining the predictors of interest and defining the inclusion and exclusion criteria of the study population. These scripts should be run in the following order:

`script_create_variables_for_vaccination_project.R`

`create_variables_from_inf_diseases_and_marriage.py`

`create_variables_for_vaccination_project_final.md`

After this, we checked the vaccination coverage in the study population and removed individuals living in on municipality with incomplete vaccination statistics, this code is in

`vacc_stats.md`

For some of the analyses, additional preprocessing is done. For training the Lasso classifier models, missing values were imputed as described in the Methods section of the manuscript. This code is in

`impute_missing_data_082022.md`

For the sensitivity analysis, we removed all individuals without records in the year 2019 (see Methods section of the manuscript for details). The script for identifying these individuals is in

`IDs_without_records_for_2019.md`

#### Machine learning analyses

The R-script for the non-penalized logistic regression analyses is in

`runLogisticRegression_chunks.R`

The R-script for the Lasso analyses is in

`runglmnet_chunks_0822.R`

The Python-script for training the XGBoost models is in

`xgboost_training_skopt.py`

Each of the scripts is written so that the model fitted can be specified by giving the full training data and a list of predictors used in the model.

#### Scripts for post-processing the results

Prevalences of the predictors were computed as in

`compute_variable_prevalences.md`

and the prevalences as well as results from the logistic regression and the Lasso analyses were combined for exporting from the secure computing environment as in

`logreg_results.md`

Analysis of the highest risk individuals according to the XGBoost model and XGBoost model re-calibration were done as in

`xgb_calibration_resuts.md`

SHAP values for the different predictors in the XGBoost model were computed as in

`compute_shap_for_xgb.md`

### Finngen - code for the genetics analyses

See the separate README.md file in the folder FinnGen-script/ for details of the gentetics analyses.
