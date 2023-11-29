# CR dynamics by rating methods and videos/quadrants
# 1. Plotting means across participants
# 2. Individual plots
# 3. Compute CR descriptive statistics
# 4. Arousal vs. Valence
#
# Author: Antonin Fourcade
# Last version: 15.08.2023

# import packages
# comment: not sure what packages are really used here
library(tidyverse)
library(ggpmisc)
library(ggdist)
library(gghalves)
library(hrbrthemes)
library(scales)
library(viridis)
library(car)
library(readxl)
library(e1071)
library(irr)

# set paths
data_path <- "E:/AffectiveVR/Data/"
save_path <- "E:/AffectiveVR/affectivevr/cr_plots/"

if (!file.exists(save_path)) {dir.create(save_path)}

# read data
data <- read.csv(paste(data_path, "cr_rs_clean.csv", sep = ""))

# set names of test sites
test_list <- c("Torino", "Berlin")
# set names of rating methods and baseline
rm_list <- c("Grid", "Flubber", "Proprioceptive")
base_name <- c("Baseline")
# set names of quadrants/videos
quad_list <- c("HP", "LP", "LN", "HN")
# set dimensions
dimensions <- c("v", "a", "dist", "angle")
dimensions_names <- str_replace_all(dimensions, c("angle" = "Angle", "a" = "Arousal", "v" = "Valence", "dist" = "Distance"))

# format data
data <- data %>% 
  mutate(
    test_site = factor(test_site, levels = test_list),
    rating_method = factor(rating_method, levels = c(rm_list, base_name)),
    quadrant = factor(quadrant, levels = quad_list),
    sj_id = factor(sj_id, levels = unique(data$sj_id)),
    cr_time = factor(cr_time, levels = unique(data$cr_time)))

# Count non-NaN datapoints per metric
colSums(!is.na(data))

# Create dataframe with mean and std for each quadrant and rating method
dataSummary <- data  %>% 
  group_by(cr_time, quadrant, rating_method) %>% 
  dplyr::summarise(n = n(),
                   cr_v_mean = mean(cr_v, na.rm = T),
                   cr_v_sd = sd(cr_v, na.rm = T),
                   cr_v_se = cr_v_sd/sqrt(n),
                   cr_a_mean = mean(cr_a, na.rm = T),
                   cr_a_sd = sd(cr_a, na.rm = T),
                   cr_a_se = cr_a_sd/sqrt(n),
                   cr_dist_mean = mean(cr_dist, na.rm = T),
                   cr_dist_sd = sd(cr_dist, na.rm = T),
                   cr_dist_se = cr_dist_sd/sqrt(n),
                   cr_angle_mean = mean(cr_angle, na.rm = T),
                   cr_angle_sd = sd(cr_angle, na.rm = T),
                   cr_angle_se = cr_angle_sd/sqrt(n))
# rename quadrants
  dataSummary$quadrant <- str_replace(dataSummary$quadrant, 'HP', 'High Arousal Positive Valence')
  dataSummary$quadrant <- str_replace(dataSummary$quadrant, 'HN', 'High Arousal Negative Valence')
  dataSummary$quadrant <- str_replace(dataSummary$quadrant, 'LP', 'Low Arousal Positive Valence')
  dataSummary$quadrant <- str_replace(dataSummary$quadrant, 'LN', 'Low Arousal Negative Valence')

# Mean across participants  ------------------------------------------------
# Plot the mean CR time-series across SJs, for each quadrant and each rating_method
# -> one figure for each dimensions (v, a , dist, angle)
  
# change names of mean measurements
mean_names <- c('Mean Valence', 'Mean Arousal', 'Mean Distance', 'Mean Angle')
names(dataSummary) <- str_replace_all(names(dataSummary), c("cr_v_mean" = mean_names[1], "cr_a_mean" = mean_names[2], "cr_dist_mean" = mean_names[3], "cr_angle_mean" = mean_names[4]))

# run a loop over all means of the four different dimensions (valence, arousal, distance, angle)
for (d in mean_names){
  # find index of dimension mean in dataSummary
  p <- match(d, names(dataSummary))
  # create a df with only one of the four continuous measurements
  mean_df <- dataSummary %>%
    dplyr::select('cr_time','quadrant','rating_method',p) %>%
    group_by(quadrant,cr_time, rating_method)
  
  # save the dimension mean as title for figure 
  title <- d
  # rename name of mean for easier plotting
  names(mean_df) <- str_replace(names(mean_df),d,'cr')
  
  # create a plot that shows the average rating for each rating method, condition, and measurement.
  # Define the y limits depending on dimension
  lim_y <- case_when(d == 'Mean Valence' ~ c(-1,1),
                     d == 'Mean Arousal' ~ c(-1,1),
                     d == 'Mean Distance' ~ c(-sqrt(2),sqrt(2)),
                     d == 'Mean Angle' ~ c(-pi,pi))
  
  plot <- ggplot(mean_df, aes(x = cr_time, y = cr)) +
    geom_line(aes(color = rating_method, group = rating_method), linewidth = 1) +
    facet_wrap(.~quadrant) +
    scale_x_discrete(breaks = seq(5, 60, 10)) +
    theme_bw(base_size = 14) +
    ggtitle(paste(title,'by Rating Method and Video/Quadrant')) +
    ylim(lim_y) +
    xlab('Time (s)') +
    ylab(paste('Continuous Rating ', title)) +
    labs(color = 'Rating Method')
   
  # save the file as vector-based graphic with type of measurement in the name
   ggsave(file= file.path(save_path, paste(sub(' ', '_',title),'CR.png',sep='_')), plot = plot, height=210,  width= 297, units = 'mm')
   
}


# Individual plots - CR time-series per rating methods and videos/ --------

# loop over every participant
sjs <- unique(data$sj_id)
for (i in sjs){
  
  # create a new df for that participant
  df_sj <- data %>%
    filter(sj_id == i)%>%
    group_by(cr_time, quadrant, rating_method) 
  
  # exclude participants with no data
  if (nrow(df_sj) == 0) next # skip itteration if df for one participant is empty (e.g. subject no 3)
  
  # create file path for every participant so save corresponding plots
  dDir <- file.path(save_path, 'individual_plots',i)
  if (!file.exists(dDir)) {dir.create(dDir)}
  
  # loop over the four dimensions (valence, arousal, distance, angle) for every participant
  for (d in 1:length(dimensions)){
    # find index of dimension cr in df_sj
    p <- match(paste0('cr_',dimensions[d]), names(df_sj))
    # create a new df for one measurement 
    df_sj_dim <- df_sj %>%
      dplyr::select('cr_time','quadrant','rating_method',p) %>%
      group_by(quadrant,cr_time, rating_method)
   
    # get title for figure
    title <- dimensions_names[d]
    # change variable name for easier plotting
    names(df_sj_dim) <- str_replace(names(df_sj_dim),paste0('cr_',dimensions[d]),'cr')
    
    # Define the y limits depending on dimension
    lim_y <- case_when(dimensions[d] == 'v' ~ c(-1,1),
                       dimensions[d] == 'a' ~ c(-1,1),
                       dimensions[d] == 'dist' ~ c(-sqrt(2),sqrt(2)),
                       dimensions[d] == 'angle' ~ c(-pi,pi))
    
    # plot rating method, for every condition across time
    plot <- ggplot(df_sj_dim, aes(x = cr_time, y = cr)) +
      geom_line(aes(color = rating_method, group = rating_method), size = 1) +
      facet_wrap(.~quadrant) +
      ylim(lim_y) +
      scale_x_discrete(breaks = seq(5, 60, 10)) +
      theme_bw(base_size = 14)+
      ggtitle(paste(title,'Continuous Ratings by Rating Method and Video/Quadrant'))+
      xlab('Time (s)')+
      ylab(paste(title, 'Continuous Ratings'))+
      labs(color = 'Rating Method')
    
    #save file
    ggsave(file= file.path(dDir, paste(title,".png", sep ='')), plot = plot, height=210,  width= 297, units = 'mm')
    
  }
  
}


# Summary descriptive statistics ------------------------------------------

# create a summary dataframe of mean, sd, skewness and kurtosis for every rating method and dimension and per quadrant (condition)
digit <- 2 # round to 'digit' digits

descriptiveSummary <- data  %>% 
  group_by(quadrant, rating_method, test_site) %>% #!grouped by test_site for following plot
  pivot_longer(c(cr_v:cr_angle)) %>% 
  group_by(rating_method , quadrant, test_site, name) %>% 
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
write_csv(descriptiveSummary, file.path(save_path,"cr_descriptive_summary.csv"))

# Relationship between valence and arousal --------------------------------

# Mean across subjects
# Separated by testing sites
plot <- ggplot(descriptiveSummary, aes(x = cr_v_mean, y = cr_a_mean, color = rating_method, shape = test_site)) +
  geom_point(size = 2) +
  # geom_errorbar(aes(
  #   ymin = cr_a_mean - cr_a_sd,
  #   ymax = cr_a_mean + cr_a_sd,
  #   x = cr_v_mean,
  #   width = 0.1
  # )) +
  # geom_errorbarh(aes(
  #   xmin = cr_v_mean - cr_v_sd,
  #   xmax = cr_v_mean + cr_v_sd,
  #   y = cr_a_mean,
  #   height = 0.1
  # )) +
  theme_bw(base_size = 14) +
  ggtitle('Mean across participants - Arousal vs. Valence') +
  xlab('Mean Valence') +
  ylab('Mean Arousal') +
  labs(color = 'Rating Method', shape = 'Testing Site') +
  ylim(c(-1,1)) +
  xlim(c(-1,1)) +
  geom_vline(xintercept = 0, color = "black") +
  geom_hline(yintercept = 0, color = "black")
plot
#save file
ggsave(file= file.path(save_path, "mean_valence_arousal_test_sites.png"), plot = plot, height=210,  width= 297, units = 'mm')

# All data points
# create dataframe for 'additive' colors
toPlot <- data %>% 
  select(rating_method, cr_v, cr_a) %>% 
  mutate_if(is.numeric, ~round(., digit)) %>% # round values
  mutate(coordinates = paste0(cr_v, ",", cr_a)) %>% # get coordinates from cr_v and cr_a
  select(-c(cr_v, cr_a)) %>% 
  group_by(rating_method, coordinates) %>% 
  mutate(n = n()) %>% # count number of cases (i.e., how many times a coordinate is hit)
  distinct() %>% # only unique rows
  pivot_wider(names_from = rating_method, values_from = n) %>% # create variables from each rating method containing the number of cases
  mutate(ratings = paste0(ifelse(!is.na(Grid), "Grid/", ""), 
                          ifelse(!is.na(Flubber), "Flubber/", ""), 
                          ifelse(!is.na(Proprioceptive), "Proprioceptive", "")), # paste together names of ratings that have non-zero number of ratings
         ratings = case_when(str_sub(ratings, -1, -1) == "/" ~ str_sub(ratings, 1, -2), # drop last slash if necessary
                             T ~ ratings), # otherwise keep rating name
         nRatings = sum(c(Grid, Flubber, Proprioceptive), na.rm = T)) %>% # tally total number of ratings across rating methods
  select(ratings, coordinates, nRatings) %>% 
  separate(coordinates, c("cr_v", "cr_a"), sep = ",") %>% # separate coordinates into cr_v and cr_a again
  mutate(across(c(cr_v, cr_a), as.numeric),
         ratings = factor(ratings, levels = c("Grid", "Flubber", "Proprioceptive",
                                              "Grid/Flubber", "Grid/Proprioceptive", "Flubber/Proprioceptive",
                                              "Grid/Flubber/Proprioceptive"))) %>% 
  uncount(nRatings) # create nRatings number of identical rows for each case

# plot dataframe for 'additive' colors
plot <- ggplot(toPlot, aes(x = cr_v, y = cr_a)) +
  geom_point(aes(color = ratings), size = 0.5, alpha = 0.1) +
  theme_bw(base_size = 14) +
  guides(color = guide_legend(override.aes = list(size = 2, alpha = 1))) +
  scale_color_manual(values = c(hue_pal()(3), "#7C9853", "#AD89B6", "#31AB9C", "#5C2C26")) +
  ggtitle('Arousal vs. Valence') +
  xlab('Valence') +
  ylab('Arousal') +
  labs(color = 'Rating Method')
plot
#save file
ggsave(file= file.path(save_path, "valence_arousal.png"), plot = plot, height=210,  width= 297, units = 'mm')

# descriptiveSummary2 <- data  %>% 
#   mutate(quadrant = factor(quadrant, levels = c("HN", "HP", "LN", "LP"))) %>% 
#   group_by(quadrant, rating_method, sj_id) %>% 
#   dplyr::summarise(n = n(),
#                    cr_v_mean = round(mean(cr_v, na.rm = T), digits = digit),
#                    cr_v_sd = round(sd(cr_v, na.rm = T), digits = digit),
#                    cr_v_skew = round(skewness(cr_v, na.rm = T, type = 2), digits = digit),
#                    cr_v_kurt = round(kurtosis(cr_v, na.rm = T, type = 2), digits = digit),
#                    cr_a_mean = round(mean(cr_a, na.rm = T), digits = digit),
#                    cr_a_sd = round(sd(cr_a, na.rm = T), digits = digit),
#                    cr_a_skew = round(skewness(cr_a, na.rm = T, type = 2), digits = digit),
#                    cr_a_kurt = round(kurtosis(cr_a, na.rm = T, type = 2), digits = digit),
#                    cr_dist_mean = round(mean(cr_dist, na.rm = T), digits = digit),
#                    cr_dist_sd = round(sd(cr_dist, na.rm = T), digits = digit),
#                    cr_dist_skew = round(skewness(cr_dist, na.rm = T, type = 2), digits = digit),
#                    cr_dist_kurt = round(kurtosis(cr_dist, na.rm = T, type = 2), digits = digit),
#                    cr_angle_mean = round(mean(cr_angle, na.rm = T), digits = digit),
#                    cr_angle_sd = round(sd(cr_angle, na.rm = T), digits = digit),
#                    cr_angle_skew = round(skewness(cr_angle, na.rm = T, type = 2), digits = digit),
#                    cr_angle_kurt = round(kurtosis(cr_angle, na.rm = T, type = 2), digits = digit)) 
# 
# plot <- ggplot(descriptiveSummary2, aes(x = cr_v_mean, y = cr_a_mean)) +
#   geom_point(aes(color = rating_method), size = 0.5) +
#   theme_bw(base_size = 14) +
#   ggtitle('Mean per SJ - Arousal vs. Valence') +
#   xlab('Mean Valence') +
#   ylab('Mean Arousal') +
#   labs(color = 'Rating Method')
# plot
# #save file
# ggsave(file= file.path(save_path, "sj_mean_valence_arousal.png"), plot = plot, height=210,  width= 297, units = 'mm')
