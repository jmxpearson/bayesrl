"""
Runs a specified Bayesian RL model on an input csv file.
For usage, do python runmodel.py --help.
"""
from __future__ import division
import numpy as np
import pandas as pd
import pystan
import sys
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a Bayesian hierarchical model")
    parser.add_argument("model", help="name of model file")
    parser.add_argument("input", help="input file name in csv format")
    parser.add_argument("-o", "--output", help="input file name in csv format; defaults to results.xlsx", default="results.xlsx")

    parser.add_argument("-s", "--seed", help="random seed for simulation",
        type=int, default=77752)
    args = parser.parse_args()

    # be nice to users by adding extensions
    if args.model.split('.')[-1] != 'stan':
        modfile = args.model + '.stan'
    else:
        modfile = args.model
    if args.input.split('.')[-1] != 'csv':
        infile = args.input + '.csv'
    else:
        infile = args.input
    if args.output.split('.')[-1] != 'xlsx':
        outfile = args.output + '.xlsx'
    else:
        outfile = args.output

    seed = args.seed

    # read in
    try:
        df = pd.read_csv(infile)
    except:
        print("Sorry, can't find {}".format(infile))
        sys.exit(1)

    # make a data dictionary to be read by Stan
    ddict = {}
    ddict['N'] = df.shape[0]
    ddict['Nsub'] = len(df['SubjNum'].unique())
    ddict['Ncue'] = sum(~np.isnan(df['Chosen'].unique()))
    ddict['Ntrial'] = np.max(df['Trial'])
    ddict['Ngroup'] = len(df['AgeGroup'].unique())
    ddict['Ncond'] = len(df['DelayCond'].unique())
    ddict['Nrun'] = len(df['RunNum'].unique())
    ddict['sub'] = df['SubjNum']
    ddict['chosen'] = df['Chosen'].fillna(0).astype('int')
    ddict['unchosen'] = df['Unchosen'].fillna(0).astype('int')
    ddict['trial'] = df['Trial']
    ddict['outcome'] = df['Outcome'].fillna(-1).astype('int')
    ddict['group'] = df[['AgeGroup', 'SubjNum']].drop_duplicates()['AgeGroup']
    ddict['condition'] = df['DelayCond'].fillna(0).astype('int')
    ddict['run'] = df['RunNum'].fillna(0).astype('int')

    # # compile stan model
    np.random.seed(seed)
    sm = pystan.StanModel(file=modfile)

    # run it
    fit = sm.sampling(data=ddict, chains=2)

    # extract samples
    samples = fit.extract()

    # prepare variables to write out
    D = np.median(samples['Delta'], 0)  # prediction error
    Q = np.median(samples['Q'], 0)  # expected value/Q-value
    sub_alpha = np.median(samples['alpha'], 0)
    if 'beta' in samples:
        sub_beta = np.median(samples['beta'], 0)
    else:
        sub_beta = None

    with pd.ExcelWriter(outfile) as writer:
        for sub in range(ddict['Nsub']):
            print("Writing subject {}".format(sub))
            df = pd.DataFrame(D[sub])
            df.to_excel(writer, sheet_name='RPE_Subject' + str(sub))
            df = pd.DataFrame(Q[sub])
            df.to_excel(writer, sheet_name='EV_Subject' + str(sub))

        df = pd.DataFrame(sub_alpha)
        df.to_excel(writer, sheet_name='Learning Rates')
        if sub_beta is not None:
            df_beta = pd.DataFrame(sub_beta)
            df_beta.to_excel(writer, sheet_name='Softmax Parameters')

        df = pd.DataFrame(samples['log_lik'])
        df.to_excel(writer, sheet_name='Log posterior samples')

    try:
        alphas = samples['alpha_pred']
        dims = alphas.shape
        grpnames = ['Younger', 'Older']
        condnames = ['Condition1', 'Condition2']

        # now figure out what variables we included in alpha
        if len(dims) > 2:
            ngroups = dims[1]
            grps = grpnames[:ngroups]
            nconds = dims[2]  # can be condition or run
            conds = condnames[:nconds]
            preds = pd.Panel(alphas, major_axis=grps, minor_axis=conds)
            preds = preds.to_frame().transpose()
        elif len(dims) > 1:
            ngroups = dims[1]
            grps = grpnames[:ngroups]
            preds = pd.DataFrame(alphas, columns=grps)
        else:
            preds = pd.Series(alphas)

        preds.to_csv('Model_preds.csv')
    except:
        print("Sorry, but there was an error writing the model predictions file.")
