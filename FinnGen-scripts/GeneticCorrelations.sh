endpoints=(ADHD_2017 anxiety anorexia_2019 autism_2017.ipsych.pgc BMI breast_cancer CAD cannabis_ever_2018.no23andMe height_combined loneliness.rapid_UKB.sumstats Neuroticism_Full openness prostate_cancer risk_behavior smoking_ever_vs_never T2D_2018 MHQPart COVID_B2 COVID_C2 education SCZ mdd BIP)
sumstat_directory="file/path/to/sumstats"
output="file/path/to/output"

touch genetic_correlations

#Run every combination of genetic correlation
for i in ${!endpoints[@]}; do

#Select that phenotype
pheno_i=${endpoints[i]}

python2 ldsc.py \
--rg ${sumstat_directory}/COVIDVax.LDSC.sumstats.gz, ${sumstat_directory}/${pheno_i}.LDSC.sumstats.gz \
--ref-ld-chr eur_w_ld_chr/ \
--w-ld-chr eur_w_ld_chr/ \
--out ${output}/covid_vax_vs_${pheno_i}

#Extract genetic correlation and standard error and append
correlation=$(grep "Genetic Correlation:" ${output}/covid_vax_vs_${pheno_i}.log|awk '{printf $3 " " $4}')

echo covid_vax_vs_${pheno_i} ": " ${correlation} >> genetic_correlations
done
