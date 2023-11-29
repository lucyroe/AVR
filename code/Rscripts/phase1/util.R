# Various functions useful for AVR analyses
#
# Author: Antonin Fourcade
# Last version: 28.06.2023

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

# plot_rm_q ------------------------------------------------------------------
plot_rm_q <- function(df, x, y, fill, xlab, ylab, breaks_y, savepath) {
  # Function to generate a customized plot using ggplot2 for questionnaires by rating methods
  # Parameters:
  # - df: Data frame containing the data for the plot
  # - x: Variable representing the x-axis
  # - y: Variable representing the y-axis
  # - fill: Variable used for fill color
  # - xlab: Label for the x-axis
  # - ylab: Label for the y-axis
  # - breaks_y: Numeric vector specifying the breaks on the y-axis
  # - savepath: File path where the plot will be saved
    # Create the ggplot object
  p <- ggplot(df, aes(x = {{x}}, y = {{y}}, fill = {{fill}})) +
    # Add a half violin plot
    geom_half_violin(position = position_nudge(x = 0.2), alpha = 0.5, side = "r") +
    # Add jittered points
    geom_point(aes(color = as.numeric(as.character(sj_id))), position = position_jitter(width = .1, height = 0), size = 1.5) +
    # Add a boxplot
    geom_boxplot(width = .1, alpha = 0.5) +
    # Set the y-axis labels and breaks
    scale_y_continuous(name = ylab, breaks = breaks_y) +
    # Set the x-axis label
    xlab(xlab) +
    # Set the color scale legend
    scale_color_continuous(name = "SJ_id") +
    # Set the theme to a white background with a base font size of 14
    theme_bw(base_size = 14)
  
  # Save the plot as an image
  ggsave(savepath, p, height = 6, width = 6)
  
  # Return the ggplot object
  return(p)
}


# Comparison Rating methods for questionnaires -----------------------------------------------
compare_rm_q <- function(df, dv, rm_list, base_name, sesoi, sesoi_type, mu, alpha, mc_cor, digit, savepath, save_suffix) {
  # This function performs various statistical analyses to compare rating methods using repeated measures. 
  # It conducts ANOVA tests and post-hoc comparisons using TOST (two one-sided t-tests) to evaluate equivalence between rating methods. 
  # The results are saved in a CSV file named "TOST.csv" and a text file named "results_compare_rm.txt".
  # Parameters:
  # - df: Data frame containing the data.
  # - dv: Name of the dependent variable column in the data frame.
  # - rm_list: List of rating methods to compare (e.g., c("RM1", "RM2", "RM3")).
  # - base_name: Name of the Baseline rating method to compare against.
  # - sesoi: Smallest effect size of interest.
  # - sesoi_type: Type of equivalence bound for TOST (e.g., "raw").
  # - mu: Null value for equivalence testing.
  # - alpha: Significance level for hypothesis testing and p-value adjustment.
  # - mc_cor: Multiple comparison correction method (e.g., "fdr", holm", "bonferroni").
  # - digit: integer indicating the number of decimal places for roundinf numeric values
  # - savepath: Directory path to save the output files.
  # - save_suffix: Name of the suffix for saved file
  
  # rename dv column for easier processing
  names(df) <- str_replace(names(df), dv, 'dv')
  
  # ANOVA with factor rating_method and three levels (RM1, RM2, RM3, omit Baseline)
  df.anova <- anova_test(df %>% filter(rating_method != base_name), dv = dv, wid = sj_id, within = rating_method, effect.size = "pes")
  get_anova_table(df.anova)
  
  # ANOVA with factor rating_method and 4 levels (RM1, RM2, RM3, Baseline)
  df_wbase.anova <- anova_test(df, dv = dv, wid = sj_id, within = rating_method, effect.size = "pes")
  get_anova_table(df_wbase.anova)
  
  ### Post-hoc comparisons
  
  # Extract data for each rating method
  rm1 <- (df %>% filter(rating_method == rm_list[1]))$dv
  rm2 <- (df %>% filter(rating_method == rm_list[2]))$dv
  rm3 <- (df %>% filter(rating_method == rm_list[3]))$dv
  baseline <- (df %>% filter(rating_method == base_name))$dv
  
  # Paired t-tests & Equivalence testing - TOST approach
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
  rm1_rm2_test <- tsum_TOST(m1=mean(rm1), sd1=sd(rm1), n1=length(rm1),
                              m2=mean(rm2), sd2=sd(rm2),  n2=length(rm2), 
                              r12 = cor(rm1, rm2),
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
  rm1_rm3_test <- tsum_TOST(m1=mean(rm1), sd1=sd(rm1), n1=length(rm1),
                                 m2=mean(rm3), sd2=sd(rm3), n2=length(rm3), 
                                 r12 = cor(rm1, rm3),
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
  rm2_rm3_test <- tsum_TOST(m1=mean(rm2), sd1=sd(rm2), n1=length(rm2),
                                 m2=mean(rm3), sd2=sd(rm3), n2=length(rm3), 
                                 r12 = cor(rm2,rm3),
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
  p_values = c(rm1_base_test$TOST$p.value[1], rm2_base_test$TOST$p.value[1], rm3_base_test$TOST$p.value[1], rm1_rm2_test$TOST$p.value[1], rm1_rm3_test$TOST$p.value[1], rm2_rm3_test$TOST$p.value[1])
  adjusted_p_values <- round(p.adjust(p_values, mc_cor), digit)
  
  # create dataframe of TOST results
  all_tosts <- data.frame(rm1_base_test$TOST, rm2_base_test$TOST, rm3_base_test$TOST, rm1_rm2_test$TOST, rm1_rm3_test$TOST, rm2_rm3_test$TOST)%>% 
    mutate_if(is.numeric, ~round(., digit))
  # add NHST FDR corrected p-values
  for (i in seq(6, 1, -1)) {
    new_column <- data.frame("FDR adjusted p-value" = c(adjusted_p_values[i],"",""))
    all_tosts <- add_column(all_tosts, new_column, .after = 4*i)
  }
  # add header with post-hoc comparisons
  all_tosts_names <- names(all_tosts)
  header <- c(paste(rm_list[1],'vs',base_name),'','','','',paste(rm_list[2],'vs',base_name),'','','','',paste(rm_list[3],'vs',base_name),'','','','',paste(rm_list[1],'vs',rm_list[2]),'','','','',paste(rm_list[1],'vs',rm_list[3]),'','','','',paste(rm_list[2],'vs',rm_list[3]),'','','','')
  names(all_tosts) <- header
  all_tosts <- rbind(all_tosts_names, all_tosts)
  
  # save dataframe in csv file
  write.csv(all_tosts, file = file.path(savepath, paste0("TOST_", save_suffix,".csv")))

  #save results in txt file
  sink(file = file.path(savepath, paste0("results_compare_rm_", save_suffix, ".txt")))
  cat(c("ANOVA with factor feedback and three levels:", rm_list, "\n"))
  print(get_anova_table(df.anova))
  cat("\n")
  
  cat(c("ANOVA with factor feedback and four levels:", rm_list, base_name, "\n"))
  print(get_anova_table(df_wbase.anova))
  cat("\n")
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
  
  cat(c("Comparison", rm_list[1], "vs.", rm_list[2], "\n"))
  print(rm1_rm2_test)
  cat(c('FDR adjusted NHST p-value: ',adjusted_p_values[4],'\n'))
  cat('\n')
  
  cat(c("Comparison", rm_list[1], "vs.", rm_list[3], "\n"))
  print(rm1_rm3_test)
  cat(c('FDR adjusted NHST p-value: ',adjusted_p_values[5],'\n'))
  cat('\n')
  
  cat(c("Comparison", rm_list[2], "vs.", rm_list[3], "\n"))
  print(rm2_rm3_test)
  cat(c('FDR adjusted NHST p-value: ',adjusted_p_values[6],'\n'))
  cat('\n')
  
  sink()
  
  return(all_tosts)
}


# add_p_ttests_rm - Additional post-hoc paired t-tests --------------------------------------
add_p_ttests_rm <- function(df, dv, rm_list, base_name, mc_cor, savepath) {
  # Perform paired t-tests and adjust p-values for multiple comparisons between rating methods
  # df: Data frame containing the data
  # dv: Name of the dependent variable column in the data frame
  # rm_list: List of rating methods to compare
  # base_name: Baseline rating method to compare against
  # mc_cor: Multiple comparison correction method
  # savepath: Directory path to save the output file
  
  # rename dv column for easier processing
  names(df) <- str_replace(names(df), dv, 'dv')
  
  # Extract data for each rating method
  rm1 <- (df %>% filter(rating_method == rm_list[1]))$dv
  rm2 <- (df %>% filter(rating_method == rm_list[2]))$dv
  rm3 <- (df %>% filter(rating_method == rm_list[3]))$dv
  baseline <- (df %>% filter(rating_method == base_name))$dv
  
  df.ttest_rm1_base <- t.test(rm1, baseline, paired = T)
  diff_rm1_base <- rm1 - baseline
  cohensd_rm1_base <- df.ttest_rm1_base$estimate / sd(diff_rm1_base) # cohen's d
  
  df.ttest_rm2_base <- t.test(rm2, baseline, paired = T)
  diff_rm2_base <- rm2 - baseline
  cohensd_rm2_base <- df.ttest_rm2_base$estimate / sd(diff_rm2_base)
  
  df.ttest_rm3_base <- t.test(rm3, baseline, paired = T)
  diff_rm3_base <- rm3 - baseline
  cohensd_rm3_base <- df.ttest_rm3_base$estimate / sd(diff_rm3_base)
  
  # Paired t-tests between each Feedback (rm1, rm2, rm3)
  df.ttest_rm1_rm2 <- t.test(rm1, rm2, paired = T)
  diff_rm1_rm2 <- rm1 - rm2
  cohensd_rm1_rm2 <- df.ttest_rm1_rm2$estimate / sd(diff_rm1_rm2)
  
  df.ttest_rm1_rm3 <- t.test(rm1, rm3, paired = T)
  diff_rm1_rm3 <- rm1 - rm3
  cohensd_rm1_rm3 <- df.ttest_rm1_rm3$estimate / sd(diff_rm1_rm3)
  
  df.ttest_rm2_rm3 <- t.test(rm2, rm3, paired = T)
  diff_rm2_rm3 <- rm2 - rm3
  cohensd_rm2_rm3 <- df.ttest_rm2_rm3$estimate / sd(diff_rm2_rm3)
  
  #extract p-value and adjust for multiple comparison
  p_values = c(df.ttest_rm1_base$p.value, df.ttest_rm2_base$p.value,df.ttest_rm3_base$p.value, df.ttest_rm1_rm2$p.value, df.ttest_rm1_rm3$p.value,df.ttest_rm2_rm3$p.value)
  adjusted_p_values <- p.adjust(p_values, mc_cor)
  
  #save results in txt file
  sink(file = file.path(savepath, "more_results_satisfaction.txt"))

  cat(c("Paired t-tests between", rm_list[1], "and", base_name, "\n"))
  print(df.ttest_rm1_base)
  cat('adjusted p-value:\n')
  print(adjusted_p_values[1])
  cat('Cohens d:\n')
  print(cohensd_rm1_base)
  cat('\n')
  
  cat(c("Paired t-tests between", rm_list[2], "and", base_name, "\n"))
  print(df.ttest_rm2_base)
  cat('adjusted p-value:\n')
  print(adjusted_p_values[2])
  cat('Cohens d:\n')
  print(cohensd_rm2_base)
  cat('\n')
  
  cat(c("Paired t-tests between", rm_list[3], "and", base_name, "\n"))
  print(df.ttest_rm3_base)
  cat('adjusted p-value:\n')
  print(adjusted_p_values[3])
  cat('Cohens d:\n')
  print(cohensd_rm3_base)
  cat('\n')
  
  cat(c("Paired t-tests between", rm_list[1], "and", rm_list[2], "\n"))
  print(df.ttest_rm1_rm2)
  cat('adjusted p-value:\n')
  print(adjusted_p_values[4])
  cat('Cohens d:\n')
  print(cohensd_rm1_rm2)
  cat('\n')
  
  cat(c("Paired t-tests between", rm_list[1], "and", rm_list[3], "\n"))
  print(df.ttest_rm1_rm3)
  cat('adjusted p-value:\n')
  print(adjusted_p_values[5])
  cat('Cohens d:\n')
  print(cohensd_rm1_rm3)
  cat('\n')
  
  cat(c("Paired t-tests between", rm_list[2], "and", rm_list[3], "\n"))
  print(df.ttest_rm2_rm3)
  cat('adjusted p-value:\n')
  print(adjusted_p_values[6])
  cat('Cohens d:\n')
  print(cohensd_rm2_rm3)
  cat('\n')
  
  sink()

}


