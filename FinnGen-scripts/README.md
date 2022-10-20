#Scripts

* Genome Wide Association Study (GWAS) using REGENIE 
  * Forms part of the pipeline embedded within FinnGen sandbox. Refer to https://finngen.gitbook.io/finngen-analyst-handbook/working-in-the-sandbox/running-analyses-in-sandbox/how-to-run-genome-wide-association-studies-gwas/how-to-run-gwas-using-regenie#example-files-for-the-regenie-pipeline/ to understand how to run REGENIE within the FinnGen infrastucture. 
  
* Genetic Correlations using Linkage Disequiblibrium Score Regression 
  * Run GeneticCorrelations.sh 
  * Munging steps have been omitted, however, we follow the default steps suggested in the wiki: https://github.com/bulik/ldsc/wiki/Heritability-and-Genetic-Correlation
  
* Heritability using Linkage Disequilibrium Score Regression
  * Run Heritability.sh
  * Munging steps have been omitted, however, we follow the default steps suggested in the wiki: https://github.com/bulik/ldsc/wiki/Heritability-and-Genetic-Correlation
  
* Meta-analysis using METAL
  * To run METAL a script is provided as input to the bash. 
  * First create a script similar to meta-analysis-script.txt. 
  * Second, feed this script as an argument into metal-bash-script.sh. 

* Mendelian Randomization using MRBase

* Polygenic Risk Scores using PRS-CS 
  * As with GWAS, PRS analysis with PRS-CS has a pipeline within the FinnGen infrastructure. Please refer to https://finngen.gitbook.io/finngen-analyst-handbook/working-in-the-sandbox/running-analyses-in-sandbox/how-to-run-prs to understad how to run PRS-CS within the FinnGen infrastructure.
