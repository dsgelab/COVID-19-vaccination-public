library(TwoSampleMR)
library(data.table)

##Read exposure data using MRBase - B2 excluding 23andMe and FinnGen
severity_exp_dat <- read_exposure_data(filename = "zcat < /Users/jermy/Downloads/COVID19_HGI_B2_ALL_leave_23andme_and_FinnGen_20220403.tsv.gz",
                                       sep = "\t",
                                       snp_col = "rsid",
                                       beta_col = "all_inv_var_meta_beta",
                                       se_col = "all_inv_var_meta_sebeta",
                                       effect_allele_col = "ALT",
                                       other_allele_col = "REF",
                                       pval_col = "all_inv_var_meta_p",
                                       chr_col="#CHR",
                                       pos_col="POS",
                                       eaf="all_meta_AF",
                                       samplesize_col="all_inv_var_meta_effective")

#Subset to significant SNPs - 5325 SNPs
severity_exp_dat <- subset(severity_exp_dat, pval.exposure <= 5e-8)

#Clumping drops from 5325 SNPs to 38
severity_exp_dat <- clump_data(severity_exp_dat)

#Outcome Data Formatting - 35 SNPs of 38 retained within the outcome dataset

##Read in summary statistics
outcome_dat <- read_outcome_data(
  snps = severity_exp_dat$SNP,
  filename = "zcat < /Users/jermy/Documents/COVID_VAX/Results/COVIDVaxwRSID.gz",
  sep = "\t",
  snp_col = "ID",
  beta_col = "BETA",
  se_col = "SE",
  effect_allele_col = "ALLELE1",
  other_allele_col = "ALLELE0",
  pval_col = "P",
  chr_col="CHROM",
  pos_col="GENPOS",
  eaf="A1FREQ",
  samplesize_col="N"
)

#Harmonize Data - Drops down to 31 SNPs due to ambiguity
dat <- harmonise_data(
  exposure_dat = severity_exp_dat, 
  outcome_dat = outcome_dat
)

#Perform MR
res <- mr(dat)
mr_pleiotropy_test(dat)
write.csv(res, "/Users/jermy/Documents/COVID_VAX/Results/MendRandResultsCOVID19SeveritycausingCOVIDVax.csv")

png("/Users/jermy/Documents/COVID_VAX/Results/MRSeverityCausingVaccination.png", height=4, width=4, units="in", res=300)
print(mr_scatter_plot(res, dat))
dev.off()