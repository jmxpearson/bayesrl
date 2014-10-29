"""
Clean memory task data and prepare for use in Stan model.
"""
from __future__ import division
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # read in data
    df = pd.read_excel("data_aug.xlsx")
    df = df.sort(['SubjNum', 'RunNum', 'TrialNum'])

    # take only the columns we want
    to_keep = ['SubjNum', 'AgeGroup', 'TrialNum', 'RunNum', 'DelayCond', 'CueLeftPic', 'CueRightPic', 
           'CueChosen', 'Outcome']
    df_red = df[to_keep].copy()

    # map cue pics to cue numbers
    df_red['CueL'] = df_red['CueLeftPic'].str[-5].astype('int')
    df_red['CueR'] = df_red['CueRightPic'].str[-5].astype('int')
    df_red['Chosen'] = df_red['CueChosen'].str[-5].astype('float')

    # make a column for unchosen option
    df_red['Unchosen'] = df_red['CueL']  # assume left option unchosen
    left_chosen = df_red['Chosen'] == df_red['CueL']  # trials when left chosen
    # now set those trials appropriately
    df_red.loc[left_chosen, 'Unchosen'] = df_red.loc[left_chosen, 'CueR']

    # don't repeat trial numbers within subject
    Ntrials = np.max(df_red['TrialNum'].unique())
    df_red['Trial'] = df_red['TrialNum'] + df_red['RunNum'] * Ntrials

    # drop redundant columns
    df_red = df_red.drop(['CueLeftPic', 'CueRightPic', 
        'CueChosen', 'CueL', 'CueR'], axis=1)

    # recode delay condition as 0/1, with 1 for delay
    df_red['DelayCond'] = df_red['DelayCond'] - 1

    # renumber subjects consecutively
    _, consec_nums = np.unique(df_red['SubjNum'], return_inverse=True)
    df_red['SubjNum'] = consec_nums + 1
    
    # write to csv
    df_red.to_csv("clean_data.csv")