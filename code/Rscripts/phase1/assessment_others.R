# Script for analysis of questionnaires in Assessment phase
# note: for invasiveness questionnaire -> separate script: invasiveness.R
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

# read assessment data
data_assess <- read.csv(paste(data_path, "assessment.csv", sep = ""))

# set names of test sites
test_list <- c("Torino", "Berlin")
# set names of rating methods and baseline
rm_list <- c("Grid", "Flubber", "Proprioceptive")
base_name <- c("Baseline")
# set names of questionnaires
q_list <- c("SUS", "invasive_presence", "Kunin")
# number of decimal places for rounding numerci values
digit <- 2

# format data
data_assess <- data_assess %>% 
  mutate(
    test_site = factor(test_site, levels = test_list),
    rating_method = factor(rating_method, levels = c(rm_list, base_name)),
    questionnaire = factor(questionnaire, levels = q_list),
    sj_id = factor(sj_id, levels = unique(data_assess$sj_id)))

# Satisfaction (Kunin scale) ----------------------------------------------

# select data
satisfactq <- filter(data_assess, questionnaire=="Kunin")

# path for saving results
Results_path_satisfact <- "E:/AffectiveVR/affectivevr/assessment_results/satisfaction/"

# plotting per rating method
p_satisfact <- plot_rm_q(df=satisfactq, x=rating_method, y=response, fill=rating_method, xlab="Rating Method", ylab="Satisfaction", breaks_y=seq(0,6,1), savepath=file.path(Results_path_satisfact, "satisfaction_rating_method.png"))
p_satisfact

# Compare ratings methods

# Set the smallest effect size of interest (SESOI)
# Here we choose 0.5 point (raw effect size) on a 7-point Likert scale
# -> ~7% change
sesoi <- c(-0.5,0.5) #c(-0.4,0.4)
sesoi_type <- c('raw')
mu <- 0 # difference in means tested
alpha <- 0.05 # alpha level
mc_cor <- "fdr"

tost_satisfact <- compare_rm_q(df=satisfactq, dv='response', rm_list=rm_list, base_name=base_name, sesoi=sesoi, sesoi_type=sesoi_type, mu=mu, alpha=alpha, mc_cor=mc_cor, digit=digit, savepath=Results_path_satisfact, save_suffix="satisfaction")

### Additional post-hoc paired t-tests
# Paired t-tests between each Feedback (Grid, Flubber, Proprioceptive) and Baseline
add_p_ttests_rm(df=satisfactq, dv='response', rm_list=rm_list, base_name=base_name, mc_cor=mc_cor, savepath=Results_path_satisfact)


# System Usability Scale (SUS) --------------------------------------------

# select data
susq <- filter(data_assess, questionnaire=="SUS")

# path for saving results
Results_path_sus <- "E:/AffectiveVR/affectivevr/assessment_results/SUS/"

# compute SUS score
# For each of the even numbered questions (positive), subtract 0 from the score.
# For each of the odd numbered questions (negative), subtract their value from 6.
# Take these new values and add up the total score. Then multiply this by 100/(7*6)=2.38
odd_q <- susq %>% filter(index %% 2 == 1) %>% group_by(sj_id, rating_method) %>% 
  dplyr::summarize(sus_score = sum(6-response))
even_q <- susq %>% filter(index %% 2 == 0) %>% group_by(sj_id, rating_method)%>% 
  dplyr::summarize(sus_score = sum(response))
sus_score <- data.frame(sj_id = odd_q$sj_id,
                        rating_method = odd_q$rating_method,
                        sus_score = (odd_q$sus_score+even_q$sus_score)*2.38)

# plotting per rating method
p_sus <- plot_rm_q(df=sus_score, x=rating_method, y=sus_score, fill=rating_method, xlab="Rating Method", ylab="SUS Score", breaks_y=seq(0,100,10), savepath=file.path(Results_path_sus, "sus_rating_method.png"))
p_sus

# Compare ratings methods

# Set the smallest effect size of interest (SESOI)
# Here we choose 5 point (raw effect size) on a 0-100 scale
# -> ~5% change
sesoi <- c(-5,5) 
sesoi_type <- c('raw')
mu <- 0 # difference in means tested
alpha <- 0.05 # alpha level
mc_cor <- "fdr"

tost_sus <- compare_rm_q(df=sus_score, dv='sus_score', rm_list=rm_list, base_name=base_name, sesoi=sesoi, sesoi_type=sesoi_type, mu=mu, alpha=alpha, mc_cor=mc_cor, digit=digit, savepath=Results_path_sus, save_suffix="sus")

# Presence and Representation (of inner emotions) ---------------------------------------------

# select data
presq <- filter(data_assess, questionnaire=="invasive_presence" & (index == 3 | index == 4))
repq <- filter(data_assess, questionnaire=="invasive_presence" & index == 2)

# path for saving results
Results_path_presrep <- "E:/AffectiveVR/affectivevr/assessment_results/presence_representation/"

# Compute Presence score
# reformat 'outside world' (negative question)
presq['response'][presq['index']==4]  <- 6 - presq['response'][presq['index']==4]
# Mean of the 2 questions
pres_score <- presq %>% group_by(sj_id, rating_method) %>% 
  dplyr::summarize(pres_score = mean(response))
pres_score <- as.data.frame(pres_score)

# plotting presence per rating method
p_pres <- plot_rm_q(df=pres_score, x=rating_method, y=pres_score, fill=rating_method, xlab="Rating Method", ylab="Presence Score", breaks_y=seq(0,6,1), savepath=file.path(Results_path_presrep, "presence_rating_method.png"))
p_pres

# plotting representation per rating method
p_rep <- plot_rm_q(df=repq, x=rating_method, y=response, fill=rating_method, xlab="Rating Method", ylab="Representation inner emotions", breaks_y=seq(0,6,1), savepath=file.path(Results_path_presrep, "representation_rating_method.png"))
p_rep

# Compare ratings methods for presence
# Set the smallest effect size of interest (SESOI)
# Here we choose 0.5 point (raw effect size) on a 7-point Likert scale
# -> ~7% change
sesoi <- c(-0.5,0.5)
sesoi_type <- c('raw')
mu <- 0 # difference in means tested
alpha <- 0.05 # alpha level
mc_cor <- "fdr"

tost_pres <- compare_rm_q(df=pres_score, dv='pres_score', rm_list=rm_list, base_name=base_name, sesoi=sesoi, sesoi_type=sesoi_type, mu=mu, alpha=alpha, mc_cor=mc_cor, digit=digit, savepath=Results_path_presrep, save_suffix="presence")

# Compare ratings methods for representation
# Set the smallest effect size of interest (SESOI)
# Here we choose 0.5 point (raw effect size) on a 7-point Likert scale
# -> ~7% change
sesoi <- c(-0.5,0.5) #c(-0.4,0.4)
sesoi_type <- c('raw')
mu <- 0 # difference in means tested
alpha <- 0.05 # alpha level
mc_cor <- "fdr"

tost_rep <- compare_rm_q(df=repq, dv='response', rm_list=rm_list, base_name=base_name, sesoi=sesoi, sesoi_type=sesoi_type, mu=mu, alpha=alpha, mc_cor=mc_cor, digit=digit, savepath=Results_path_presrep, save_suffix="representation")

