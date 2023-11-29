# Script for analysis of the rating methods' invasiveness
# 1. Invasiveness questionnaire
# 2. Summary ratings (SR) comparison between the rating methods and baseline
#      + SR descriptive statistics
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
library(rstatix)
library(TOSTER)

source("E:/AffectiveVR/affectivevr/util.R")

# set paths
data_path <- "E:/AffectiveVR/Data/"

# read data
data_sr <- read.csv(paste(data_path, "cr_sr_clean.csv", sep = ""))
data_assess <- read.csv(paste(data_path, "assessment.csv", sep = ""))

# set names of test sites
test_list <- c("Torino", "Berlin")
# set names of rating methods and baseline
rm_list <- c("Grid", "Flubber", "Proprioceptive")
base_name <- c("Baseline")
# set names of quadrants/videos
quad_list <- c("HP", "LP", "LN", "HN")
# set names of questionnaires
q_list <- c("SUS", "invasive_presence", "Kunin")
# number of decimal places for rounding numerci values
digit <- 2

# format data
data_sr <- data_sr %>% 
  mutate(
    test_site = factor(test_site, levels = test_list),
    rating_method = factor(rating_method, levels = c(rm_list, base_name)),
    quadrant = factor(quadrant, levels = quad_list),
    sj_id = factor(sj_id, levels = unique(data_sr$sj_id)))
data_assess <- data_assess %>% 
  mutate(
    test_site = factor(test_site, levels = test_list),
    rating_method = factor(rating_method, levels = c(rm_list, base_name)),
    questionnaire = factor(questionnaire, levels = q_list),
    sj_id = factor(sj_id, levels = unique(data_sr$sj_id)))

# select data
# question about invasiveness is index 1 in invasive_presence
invq <- filter(data_assess, questionnaire=="invasive_presence" & index==1)
# summary ratings (SR)
sr <- data_sr %>% dplyr::select(starts_with("sr"),"sj_id", "rating_method", "quadrant")
# SR dimensions
dimensions <- unlist(str_split(names(sr), "_"))[seq(2, length(sr)+1, 2)]
dimensions_names <- str_replace_all(dimensions, c("angle" = "Angle", "a" = "Arousal", "v" = "Valence", "dist" = "Distance"))

# Invasiveness questionnaire ----------------------------------------

# path for saving results
Results_path_invq <- "E:/AffectiveVR/affectivevr/assessment_results/invasiveness/questionnaire/"

# plotting per rating method
p_invq <- plot_rm_q(df=invq, x=rating_method, y=response, fill=rating_method, xlab="Rating Method", ylab="Invasiveness", breaks_y=seq(0,6,1), savepath=file.path(Results_path_invq, "invasiveness_rating_method.png"))
p_invq

# Compare ratings methods

# Set the smallest effect size of interest (SESOI)
# Here we choose 0.5 point (raw effect size) on a 7-point Likert scale
# -> ~7% change
sesoi <- c(-0.5,0.5) #c(-0.4,0.4)
sesoi_type <- c('raw')
mu <- 0 # difference in means tested
alpha <- 0.05 # alpha level
mc_cor <- "fdr"

tost_invq <- compare_rm_q(df=invq, dv='response', rm_list=rm_list, base_name=base_name, sesoi=sesoi, sesoi_type=sesoi_type, mu=mu, alpha=alpha, mc_cor=mc_cor, digit=digit, savepath=Results_path_invq, save_suffix='invasiveness')

# Additional post-hoc paired t-tests
# Paired t-tests between each Feedback (Grid, Flubber, Proprioceptive) and Baseline
add_p_ttests_rm(df=invq, dv='response', rm_list=rm_list, base_name=base_name, mc_cor=mc_cor, savepath=Results_path_invq)

# SR comparison between RMs -------------------------------------------------------

# Paired t-tests and equivalence testing between each RM (Grid, Flubber, Proprioceptive) and Baseline SR 
# for each dimension (Arousal, Valence, Distance, Angle) - TOST approach
Results_path_sr <- "E:/AffectiveVR/affectivevr/assessment_results/invasiveness/SR/"

# create summary matrix for TOST results
summary_tost <- vector(mode = "list", length = length(dimensions))

# Set the smallest effect size of interest (SESOI)
# Here we choose 0.125 point (raw effect size) on a [-1,1] rating for arousal and valence
# For distance: [0, -sqrt(2)] ratings -> 0.125*sqrt(2) = 0.177
# For angle: [-pi, pi] ratings -> pi*0.125 = 0.393
# -> ~6.25% change
sesoi_a_v <- c(-0.125,0.125)
sesoi_dist <- c(0,0.177)
sesoi_ang <- c(-0.393,0.393)
sesoi_type <- c('raw')
mu <- 0 # difference in means tested
alpha <- 0.05 # alpha level

for (d in 1:length(dimensions)){
  # create results folder
  curDir <- file.path(Results_path_sr, dimensions_names[d])
  if (!file.exists(curDir)) {dir.create(curDir)}
  
  #subset data
  curDim <- sr %>% 
    dplyr::select(ends_with(dimensions[d]), "sj_id", "rating_method", "quadrant")
  # rename sr_dimension into sr for easier processing
  names(curDim)[1] <- "sr"
  
  # Define the y limits depending on dimension
  lim_y <- case_when(dimensions[d] == 'v' ~ c(-1,1),
                     dimensions[d] == 'a' ~ c(-1,1),
                     dimensions[d] == 'dist' ~ c(0,sqrt(2)),
                     dimensions[d] == 'angle' ~ c(-pi,pi))
  
  # plotting per rating method
  p_rm <- ggplot(curDim, aes(x = rating_method, y = sr, fill = rating_method)) +
    geom_half_violin(position = position_nudge(x = 0.2), alpha = 0.5, side = "r") +
    geom_point(aes(color = as.numeric(as.character(sj_id))), position = position_jitter(width = .1, height = 0), size = 1.5) +
    geom_boxplot(width = .1, alpha = 0.5) +
    ylim(lim_y) +
    ylab(paste0(dimensions_names[d], " SR")) +
    xlab("Rating method") +
    scale_color_continuous(name = "SJ_id") +
    theme_bw(base_size = 14)
  p_rm
  ggsave(file.path(curDir, paste0("sr_rating_method_", dimensions_names[d] ,".png")), p_rm, height = 6, width = 8)
  
  # plotting per rating method and video/quadrant
  p_rm_quad <- ggplot(curDim, aes(x = rating_method, y = sr, fill = rating_method)) +
    geom_half_violin(position = position_nudge(x = 0.2), alpha = 0.5, side = "r") +
    geom_point(aes(color = as.numeric(as.character(sj_id))), position = position_jitter(width = .1, height = 0), size = 1.5) +
    geom_boxplot(width = .1, alpha = 0.5) +
    facet_wrap(.~quadrant) +
    theme_bw(base_size = 14) +
    ggtitle(paste0(dimensions_names[d], " SR by Rating Method and Video/Quadrant")) +
    ylim(lim_y) +
    xlab("Rating method") +
    ylab(paste0(dimensions_names[d], " SR")) +
    labs(color = 'Rating Method')
  p_rm_quad
  ggsave(file.path(curDir, paste0("sr_quadrant_rating_method_", dimensions_names[d] ,".png")), p_rm_quad, height = 6, width = 8)
  
  # Paired t-tests & Equivalence testing - TOST approach

  rm1 <- (curDim %>% filter(rating_method == rm_list[1]))$sr
  rm2 <- (curDim %>% filter(rating_method == rm_list[2]))$sr
  rm3 <- (curDim %>% filter(rating_method == rm_list[3]))$sr
  baseline <- (curDim %>% filter(rating_method == base_name))$sr
  
  sesoi <- case_when(dimensions_names[d] == 'Valence' ~ sesoi_a_v,
                     dimensions_names[d] == 'Arousal' ~ sesoi_a_v,
                     dimensions_names[d] == 'Distance' ~ sesoi_dist,
                     dimensions_names[d] == 'Angle' ~ sesoi_ang)
  
  rm1_base_test <- tsum_TOST(m1=mean(rm1), sd1=sd(rm1), n1=length(rm1),
                              m2=mean(baseline), sd2=sd(baseline),  n2=length(baseline), 
                              r12 = cor(rm1, baseline),
                              hypothesis = c("EQU"),
                              paired = T,
                              var.equal = T,
                              eqb = sesoi, 
                              mu = mu,
                              eqbound_type = sesoi_type,
                              alpha = alpha,
                              bias_correction = TRUE,
                              rm_correction = T,
                              glass = NULL,
                              smd_ci = c( "nct"))
  rm2_base_test <- tsum_TOST(m1=mean(rm2), sd1=sd(rm2), n1=length(rm2),
                              m2=mean(baseline), sd2=sd(baseline), n2=length(baseline), 
                              r12 = cor(rm2,baseline),
                              hypothesis = c("EQU"),
                              paired = T,
                              var.equal = T,
                              eqb = sesoi,
                              mu = mu,
                              eqbound_type = sesoi_type,
                              alpha = alpha,
                              bias_correction = TRUE,
                              rm_correction = T,
                              glass = NULL,
                              smd_ci = c( "nct"))
  rm3_base_test <- tsum_TOST(m1=mean(rm3), sd1=sd(rm3), n1=length(rm3),
                              m2=mean(baseline), sd2=sd(baseline), n2=length(baseline), 
                              r12 = cor(rm3, baseline),
                              hypothesis = c("EQU"),
                              paired = T,
                              var.equal = T,
                              eqb = sesoi,
                              mu = mu,
                              eqbound_type = sesoi_type,
                              alpha = alpha,
                              bias_correction = TRUE,
                              rm_correction = T,
                              glass = NULL,
                              smd_ci = c( "nct"))

  
  # extract NHST p-values and correct for multiple comparison
  p_values = c(rm1_base_test$TOST$p.value[1], rm2_base_test$TOST$p.value[1], rm3_base_test$TOST$p.value[1])
  adjusted_p_values <- round(p.adjust(p_values, 'fdr'), digit)
  
  # create dataframe of TOST results
  all_tosts <- data.frame(rm1_base_test$TOST, rm2_base_test$TOST, rm3_base_test$TOST) %>% 
    mutate_if(is.numeric, ~round(., digit))
  # add NHST FDR corrected p-values
  for (i in seq(3, 1, -1)) {
    new_column <- data.frame("FDR adjusted p-value" = c(adjusted_p_values[i],"",""))
    all_tosts <- add_column(all_tosts, new_column, .after = 4*i)
  }
  # add header with post-hoc comparisons
  all_tosts_names <- names(all_tosts)
  header <- c(paste(rm_list[1],'vs',base_name),'','','','',paste(rm_list[2],'vs',base_name),'','','','',paste(rm_list[3],'vs',base_name),'','','','')
  names(all_tosts) <- header
  all_tosts <- rbind(all_tosts_names, all_tosts)
  row.names(all_tosts)[1] <- dimensions_names[d]
  
  # save dataframe in summary matrix
  summary_tost[[d]] <- all_tosts
  
  # save dataframe in csv file
  write.csv(all_tosts, file = file.path(curDir, paste0("TOST_", dimensions_names[d], ".csv")))
  
  #save results in txt file
  sink(file = file.path(curDir, paste0("results_", dimensions_names[d],".txt")))

  cat(c("Comparison", rm_list[1], "vs.", base_name, "\n"))
  print(rm1_base_test)
  cat(c('FDR adjusted NHST p-value: ',adjusted_p_values[1],'\n'))
  cat('\n')
  
  cat(c("Comparison", rm_list[2], "vs.", base_name, "\n"))
  print(rm2_base_test)
  cat(c('FDR adjusted NHST p-value: ',adjusted_p_values[2],'\n'))
  cat('\n')
  
  cat(c("Comparison", rm_list[3], "vs.", base_name, "\n"))
  print(rm3_base_test)
  cat(c('FDR adjusted NHST p-value: ',adjusted_p_values[3],'\n'))
  cat('\n')
  
  sink()
  
}

# Reformat summary dataframe
names(summary_tost) <- dimensions_names

# save summary dataframe in csv file
file_path = file.path(Results_path_sr, 'summary_sr_tost.csv')
if (file.exists(file_path)) {file.remove(file_path)} # delete file if already exist (because append otherwise)
lapply(summary_tost, function(x) write.table(x, file = file_path, append=T, sep=',', row.names=T, col.names=T ))

# SR Summary descriptive statistics ------------------------------------------

# create a summary dataframe of mean, sd, skewness and kurtosis for every rating method and dimension and per quadrant (condition)
descriptiveSummary <- sr  %>% 
  mutate(quadrant = factor(quadrant, levels = quad_list)) %>% 
  group_by(quadrant, rating_method) %>% 
  pivot_longer(c(sr_v:sr_angle)) %>% 
  group_by(rating_method , quadrant, name) %>% 
  dplyr::summarise(mean = mean(value, na.rm = T),
                   sd = sd(value, na.rm = T),
                   skew = skewness(value, na.rm = T),
                   kurt = kurtosis(value, na.rm = T)) %>% 
  dplyr::rename(var = name) %>% 
  pivot_longer(mean:kurt) %>% 
  unite(name, c(var, name)) %>% 
  pivot_wider(names_from = name, values_from = value) %>% 
  mutate_if(is.numeric, ~round(., digit))

#save summary df
write_csv(descriptiveSummary, file.path(Results_path_sr,"sr_descriptive_summary.csv"))

