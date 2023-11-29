# CR indices (CRi) and SR - Correlation and cocor analyses
# 1. Correlation including all data
# 2. Correlation by rating method for specific CRi (metrics) and comparisons
#
# Author: Antonin Fourcade
# Last version: 15.08.2023

# import packages
library(tidyverse)
library(ggpmisc)
library(ggdist)
library(gghalves)
library(hrbrthemes)
library(viridis)
library(car)
library(readxl)
library(cocor)
library(irr)

# set paths
data_path <- "E:/AffectiveVR/Data/"

# read data
data <- read.csv(paste(data_path, "cr_sr_clean.csv", sep = ""))

# set names of test sites
test_list <- c("Torino", "Berlin")
# set names of rating methods and baseline
rm_list <- c("Grid", "Flubber", "Proprioceptive")
base_name <- c("Baseline")
# set names of quadrants/videos
quad_list <- c("HP", "LP", "LN", "HN")

# format data
data <- data %>% 
  mutate(
    test_site = factor(test_site, levels = test_list),
    rating_method = factor(rating_method, levels = c(rm_list, base_name)),
    quadrant = factor(quadrant, levels = quad_list),
    sj_id = factor(sj_id, levels = unique(data$sj_id)))

#Count non-NaN datapoints per metric
colSums(!is.na(data))

# Correlation including all data -------------------------------------------------------

corResults_path <- "E:/AffectiveVR/affectivevr/cor_results/"
if (!file.exists(corResults_path)) {dir.create(corResults_path)}
setwd(corResults_path)

sr <- data %>% dplyr::select(starts_with("sr"))
dimensions <- unlist(str_split(names(sr), "_"))[seq(2, length(sr)*2, 2)]
dimensions_names <- str_replace_all(dimensions, c("angle" = "Angle", "a" = "Arousal", "v" = "Valence", "dist" = "Distance"))

cor <- vector(mode = "list", length = ncol(sr))

for (d in 1:ncol(sr)){
  curDim <- data %>% 
    dplyr::select(ends_with(dimensions[d])) %>% 
    pivot_longer(!names(sr[d]))
  
  vars <- unique(curDim$name)
  
  for (i in 1:length(vars)){
    # subset data
    curDat <- curDim %>% filter(name == vars[i])
    names(curDat)[1] <- "sr"
    
    # create results folder
    curDir <- file.path(corResults_path, vars[i])
    if (!file.exists(curDir)) {dir.create(curDir)}
    
    # calculate correlation coef
    curCor <- cor.test(curDat$sr, curDat$value, method = "pearson")
    
    # save correlation coef
    cor[[d]][i] <- curCor$estimate
    
    # save cor results in txt file
    sink(file = file.path(curDir, paste0("results_",vars[i],".txt")))
    cat(paste("Results" , vars[i], "\n"))
    print(curCor)
    sink()
    
    
  }
}
# create dataframe of all correlation coefficients
names(cor) <- dimensions_names
cor <- data.frame(cor)
cor <- round(cor, digits = 3)
row.names(cor) <- str_remove(vars, paste0("_", dimensions[d]))

# Save cor dataframe
write.csv(cor, file.path(corResults_path, "summary_cor_results.csv"))

# Get metrics with min and max (absolute) cor for each dimension
cor_transpose <- t(abs(cor))
max_cor_metric <- colnames(cor_transpose)[apply(cor_transpose, 1, which.max)]
min_cor_metric <- colnames(cor_transpose)[apply(cor_transpose, 1, which.min)]
# create min_max_cor dataframe
min_max_cor <- rbind(max_cor_metric, min_cor_metric)
min_max_cor <- data.frame(min_max_cor)
names(min_max_cor) <- dimensions_names
# save dataframe in csv file
write.csv(min_max_cor, file.path(corResults_path, "min_max_cor.csv"))

cat(c('Metrics with min and max CRi-SR cor:\n '))
print(min_max_cor)

# Correlation by rating method for metrics: mean, std, skewness and kurtosis --------------------------------------------

cocorResults_path <- "E:/AffectiveVR/affectivevr/cocor_results"
if (!file.exists(cocorResults_path)) {dir.create(cocorResults_path)}
setwd(cocorResults_path)

# chose specific metrics to run the analysis with, and format names
metrics <- c('cr_mean', 'cr_std', 'cr_skew', 'cr_kurtosis')
metrics_names <- str_remove(metrics, "cr_")
metrics_names <- str_replace(metrics_names, "skew", "skewness")

# label cocor parameters
j.label <- paste0("CR_", rm_list[1])
k.label <- paste0("SR_", rm_list[1])
h.label <- paste0("CR_", rm_list[2])
m.label <- paste0("SR_", rm_list[2])
n.label <- paste0("CR_", rm_list[3])
q.label <- paste0("SR_", rm_list[3])
# create summary matrices to save cor and cocor results
cor_jk <- vector(mode = "list", length = ncol(sr))
cor_hm <- vector(mode = "list", length = ncol(sr))
cor_nq <- vector(mode = "list", length = ncol(sr))
cocor_jk_hm <- vector(mode = "list", length = ncol(sr))
cocor_jk_nq <- vector(mode = "list", length = ncol(sr))
cocor_hm_nq <- vector(mode = "list", length = ncol(sr))

# get number of participants
n <- length(unique(data$sj_id))

# loop to run the entire analysis by metric and dimension
for (d in 1:ncol(sr)){
  # create a file path for every dimension
  dDir <- file.path(cocorResults_path, dimensions_names[d])
  if (!file.exists(dDir)) {dir.create(dDir)}
  
  # vector of metric and dimension names
  metrics_d <- c(paste(metrics[1], dimensions[d], sep="_"), paste(metrics[2], dimensions[d], sep="_"), paste(metrics[3], dimensions[d], sep="_"), paste(metrics[4], dimensions[d], sep="_"))
  curSr <-  names(sr[d])
  for (m in 1:length(metrics_d)){
    # create results folder for every metric 
    curDir <- file.path(dDir, metrics_d[m])
    if (!file.exists(curDir)) {dir.create(curDir)}
    
    # subset data 
    curDim <- data %>% 
      dplyr::select(all_of(curSr),metrics_d[m], rating_method) 
    # separate data per rating method
    rm1 <- curDim %>% 
      filter(rating_method == rm_list[1])
    rm2 <- curDim %>% 
      filter(rating_method == rm_list[2])
    rm3 <- curDim %>% 
      filter(rating_method == rm_list[3])
    
    # Compute CRi-SR correlations and all combinations of variables
    r.jk <- cor(rm1[,metrics_d[m]], rm1[,curSr], use = "complete.obs")[1]
    r.hm <- cor(rm2[,metrics_d[m]], rm2[,curSr], use = "complete.obs")[1]
    r.nq <- cor(rm3[,metrics_d[m]], rm3[,curSr], use = "complete.obs")[1]
    
    r.jh <- cor(rm1[,metrics_d[m]], rm2[,metrics_d[m]], use = "complete.obs")[1]
    r.jm <- cor(rm1[,metrics_d[m]], rm2[,curSr], use = "complete.obs")[1]
    r.kh <- cor(rm1[,curSr], rm2[,metrics_d[m]], use = "complete.obs")[1]
    r.km <- cor(rm1[,curSr], rm2[,curSr], use = "complete.obs")[1]
    
    r.jn <- cor(rm1[,metrics_d[m]], rm3[,metrics_d[m]], use = "complete.obs")[1]
    r.jq <- cor(rm1[,metrics_d[m]], rm3[,curSr], use = "complete.obs")[1]
    r.kn <- cor(rm1[,curSr], rm3[,metrics_d[m]], use = "complete.obs")[1]
    r.kq <- cor(rm1[,curSr], rm3[,curSr], use = "complete.obs")[1]
    
    r.hn <- cor(rm2[,metrics_d[m]], rm3[,metrics_d[m]], use = "complete.obs")[1]
    r.hq <- cor(rm2[,metrics_d[m]], rm3[,curSr], use = "complete.obs")[1]
    r.mn <- cor(rm2[,curSr], rm3[,metrics_d[m]], use = "complete.obs")[1]
    r.mq <- cor(rm2[,curSr], rm3[,curSr], use = "complete.obs")[1]
  
    # Compare dependent non-overlapping correlations, using cocor package
    diff.jk.hm <- cocor.dep.groups.nonoverlap(r.jk, r.hm, r.jh, r.jm, r.kh, r.km, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, var.labels = c(j.label, k.label, h.label, m.label), return.htest = TRUE)
    diff.jk.nq <- cocor.dep.groups.nonoverlap(r.jk, r.nq, r.jn, r.jq, r.kn, r.kq, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, var.labels = c(j.label, k.label, n.label, q.label), return.htest = TRUE)
    diff.hm.nq <- cocor.dep.groups.nonoverlap(r.hm, r.nq, r.hn, r.hq, r.mn, r.mq, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, var.labels = c(h.label, m.label, n.label, q.label), return.htest = TRUE)
    
    # store results in summary matrices
    cor_jk[[d]][m] <- round(r.jk, digits = 3)
    cor_hm[[d]][m] <- round(r.hm, digits = 3)
    cor_nq[[d]][m] <- round(r.nq, digits = 3)
    cocor_jk_hm[[d]][m] <- round(diff.jk.hm$silver2004$p.value, digits = 3)
    cocor_jk_nq[[d]][m] <- round(diff.jk.nq$silver2004$p.value, digits = 3)
    cocor_hm_nq[[d]][m] <- round(diff.hm.nq$silver2004$p.value, digits = 3)
    
    # save cocor results in txt file (nicely formatted -> return.htest = FALSE)
    sink(file = file.path(curDir, paste0("cocor_results_", metrics_d[m],".txt")))
    cat(c(rm_list[1], "vs.", rm_list[2], "\n"))
    print(cocor.dep.groups.nonoverlap(r.jk, r.hm, r.jh, r.jm, r.kh, r.km, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, var.labels = c(j.label, k.label, h.label, m.label), return.htest = FALSE))
    cat('\n')
    cat(c(rm_list[1], "vs.", rm_list[3], "\n"))
    print(cocor.dep.groups.nonoverlap(r.jk, r.nq, r.jn, r.jq, r.kn, r.kq, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, var.labels = c(j.label, k.label, n.label, q.label), return.htest = FALSE))
    cat('\n')
    cat(c(rm_list[2], "vs.", rm_list[3], "\n"))
    print(cocor.dep.groups.nonoverlap(r.hm, r.nq, r.hn, r.hq, r.mn, r.mq, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, var.labels = c(h.label, m.label, n.label, q.label), return.htest = FALSE))
    cat('\n')
    sink()
 }
}

### Create dataframe summarizing all cor and cocor results
# create dfs of each summary matrix
cor_jk <- data.frame(cor_jk, row.names = metrics_names)
names(cor_jk) <- dimensions_names
cor_hm <- data.frame(cor_hm, row.names = metrics_names)
names(cor_hm) <- dimensions_names
cor_nq <- data.frame(cor_nq, row.names = metrics_names)
names(cor_nq) <- dimensions_names
cocor_jk_hm <- data.frame(cocor_jk_hm, row.names = metrics_names)
names(cocor_jk_hm) <- dimensions_names
cocor_jk_nq <- data.frame(cocor_jk_nq, row.names = metrics_names)
names(cocor_jk_nq) <- dimensions_names
cocor_hm_nq <- data.frame(cocor_hm_nq, row.names = metrics_names)
names(cocor_hm_nq) <- dimensions_names

# create df with all cocor results
cocor_res <- list(cocor_jk_hm,cocor_jk_nq,cocor_hm_nq)
cocor_list <- map_dfr(cocor_res,function(i){
  c(unlist(i[1]),unlist(i[2]),unlist(i[3]), unlist(i[4]))})
cocor_list <- mutate_all(cocor_list, ~ paste("p =",.)) 
row_names <- c(paste(rm_list[1],"vs.",rm_list[2]), paste(rm_list[1],"vs.",rm_list[3]), paste(rm_list[2],"vs.",rm_list[3]))
cocor_df <- data.frame(cocor_list, row.names = row_names)

# create df with all cor results
cor_res <- list(cor_jk, cor_hm, cor_nq)
cor_list <- map_dfr(cor_res,function(i){
  c(unlist(i[1]),unlist(i[2]),unlist(i[3]), unlist(i[4]))})
row_names <- c(paste0("r_", rm_list[1]), paste0("r_", rm_list[2]), paste0("r_", rm_list[3]))
cor_df <- data.frame(cor_list, row.names = row_names)

# merge cor and cocor df in one summary df
summary_df <- rbind(cor_df,cocor_df)
cri_row <- rep(metrics_names, length(dimensions))
summary_df <- rbind(cri_row, summary_df)
row.names(summary_df)[1] <- c('CRi')
header <- c(dimensions_names[1],'','','',dimensions_names[2],'','','',dimensions_names[3],'','','',dimensions_names[4], '','','')
names(summary_df) <- header

# save the summary df
write.csv(summary_df, file.path(cocorResults_path, 'summary_cocor_cor_results.csv'))

# ICC - Test-retest reliability for CR between the 2 testing sites --------

iccResults_path <- "E:/AffectiveVR/affectivevr/icc_results/"
if (!file.exists(iccResults_path)) {dir.create(iccResults_path)}
setwd(iccResults_path)
file_sr <- file.path(iccResults_path, "sr_icc_results.txt")
if (file.exists(file_sr)) {file.remove(file_sr)}
file_cr <- file.path(iccResults_path, "cr_icc_results.txt")
if (file.exists(file_cr)) {file.remove(file_cr)}

digit <- 2 
dataSummary <- data  %>% 
  group_by(quadrant, rating_method, test_site) %>% #!grouped by test_site for following plot
  pivot_longer(c(sr_v:cr_cp_angle)) %>% 
  group_by(rating_method , quadrant, test_site, name) %>% 
  dplyr::summarise(mean = mean(value, na.rm = T)) %>% 
  dplyr::rename(var = name) %>% 
  pivot_longer(mean) %>% 
  unite(name, c(var, name)) %>% 
  pivot_wider(names_from = name, values_from = value) %>% 
  mutate_if(is.numeric, ~round(., digit))

site1_sr <- dataSummary %>% filter(test_site==test_list[1])  %>%
  ungroup() %>% 
  dplyr::select(starts_with('sr')) 
site2_sr <- dataSummary %>% filter(test_site==test_list[2])  %>%
  ungroup() %>%
  dplyr::select(starts_with('sr'))
site1_cr <- dataSummary %>% filter(test_site==test_list[1])  %>%
  ungroup() %>% 
  dplyr::select(starts_with('cr_mean')) 
site2_cr <- dataSummary %>% filter(test_site==test_list[2])  %>%
  ungroup() %>%
  dplyr::select(starts_with('cr_mean'))

for (d in 1:length(dimensions)){
  curDim_site1_sr <- site1_sr %>% 
    dplyr::select(ends_with(paste0(dimensions[d], '_mean')))
  curDim_site2_sr <- site2_sr %>% 
    dplyr::select(ends_with(paste0(dimensions[d], '_mean')))
  curDim_sr_icc <-data.frame('site1' = curDim_site1_sr, 'site2' = curDim_site2_sr)
  # save icc results in txt file
  sink(file = file_sr, append = TRUE)
  cat(c(str_to_upper(dimensions_names[d]), "- Test-retest", test_list[1], "-", test_list[2], "\n"))
  print(icc(curDim_sr_icc, model = "twoway", type = "consistency", unit = "average"))
  cat('\n')
  sink()
  
  curDim_site1_cr <- site1_cr %>% 
    dplyr::select(ends_with(paste0(dimensions[d], '_mean')))
  curDim_site2_cr <- site2_cr %>% 
    dplyr::select(ends_with(paste0(dimensions[d], '_mean')))
  curDim_cr_icc <-data.frame('site1' = curDim_site1_cr, 'site2' = curDim_site2_cr)
  # save icc results in txt file
  sink(file = file_cr, append = TRUE)
  cat(c(str_to_upper(dimensions_names[d]), "- Test-retest", test_list[1], "-", test_list[2], "\n"))
  print(icc(curDim_cr_icc, model = "twoway", type = "consistency", unit = "average"))
  cat('\n')
  sink()
  
}

  

