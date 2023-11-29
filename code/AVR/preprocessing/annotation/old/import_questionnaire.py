########################################################################################################################
# Script to import Questionnaire data from AVR experiment
# Output: csv file (dataframe)
# Author: Antonin Fourcade
# Last version: 15.08.2023
########################################################################################################################

# import packages
import pandas as pd
import numpy as np
import os

## Set paths and experiment parameters
data_path = 'E:/AffectiveVR/Data/'
blocks = ['Practice', 'Experiment', 'Assessment']
logging_freq = ['CR', 'SR']
test_site = ['Torino', 'Berlin']  # Torino (even SJ nb) = 0, Berlin (odd SJ nb) = 1
rating_method = ['Grid', 'Flubber', 'Proprioceptive', 'Baseline']
quadrant = ['HP', 'LP', 'LN', 'HN']
questionnaire = ['SUS', 'invasive_presence', 'Kunin']
sess = 'S000'   # session of recording

debug = False  # debug mode

# get list of participants
sj_list = os.listdir(data_path + 'AVR/')
# sort participants in ascending order
sj_list.sort()
# exclude participants because no Grid CR log recorded
to_be_removed = {'01', '02', '03', '04', '05', '07', '09', '11', '13', '15', '17'}
sj_list = [item for item in sj_list if item not in to_be_removed]

# debug
if debug:
    sj_id = '19'
    rat_mtd = 'Proprioceptive'
    vid = 'HP'

# Initialize variables
sub = []
site = []
rat_m = []
quad = []
assess = []
q_nb = []
response = []

# Read and preprocess data
# Loop over participants
for sj, sj_id in enumerate(sj_list):
    # Read results in csv file
    sj_path = data_path + 'AVR/' + sj_id + '/' + sess + '/'
    trial_results_filename = sj_path + 'trial_results.csv'
    trials_results = pd.read_csv(trial_results_filename)

    # Loop over rating methods
    for rm, rat_mtd in enumerate(rating_method):
        # Select data according to rat_mtd
        trials_rat_mtd = trials_results[trials_results['method_descriptor'] == rat_mtd]
        # Select Assessment data
        trials_ass = trials_rat_mtd[trials_rat_mtd['block_name'] == 'Assessment']

        # Loop over questionnaires
        for i_q, q in enumerate(questionnaire):
            quest_loc = 'questionnaire_' + q + '_location_0'
            quest_path = data_path + trials_ass.loc[trials_ass['qtype'] == i_q+1, quest_loc].item()
            quest = pd.read_csv(quest_path)
            # Loop over responses
            for i_r, resp in enumerate(quest['response']):
                sub = np.append(sub, sj_id)
                rat_m = np.append(rat_m, rat_mtd)
                assess = np.append(assess, q)
                q_nb = np.append(q_nb, quest['index'][i_r])
                response = np.append(response, resp)
                # Sjs with even number are in the first testing site, odd in the second
                if int(sj_id) % 2 == 0:
                    site = np.append(site, test_site[0])
                else:
                    site = np.append(site, test_site[1])
    print("SJ" + sj_id + " done")  # show progress

# create and save dataframe with Questionnaire data
d = {'sj_id': sub, 'test_site': site, 'rating_method': rat_m, 'questionnaire': assess, 'index': q_nb, 'response': response}
filename = data_path + 'assessment.csv'
df = pd.DataFrame(data=d)
df.to_csv(filename, na_rep='NaN', index=False)
