"""
Runs a specified Bayesian RL model on an input csv file.
For usage, do python runmodel.py --help.

Model1, a Bayesian RL model with hierarchical group effects, on 
input csv file. Usage:
python model1.py <infile> -o <outfile>
If <outfile> is unspecified, defaults to "results.xlsx"
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
    parser.add_argument("-o", "--output", help="input file name in csv format; defaults to results.xlsx",
        default="results.xlsx")
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
        print "Sorry, can't find {}".format(infile)
        sys.exit(1)

    # make a data dictionary to be read by Stan
    ddict = {}
    ddict['N'] = df.shape[0]
    ddict['Nsub'] = len(df['SubjNum'].unique())
    ddict['Ncue'] = sum(~np.isnan(df['Chosen'].unique()))
    ddict['Ntrial'] = np.max(df['Trial'])
    ddict['Ngroup'] = len(df['AgeGroup'].unique())
    ddict['sub'] = df['SubjNum']
    ddict['chosen'] = df['Chosen'].fillna(0).astype('int')
    ddict['unchosen'] = df['Unchosen'].fillna(0).astype('int')
    ddict['trial'] = df['Trial']
    ddict['outcome'] = df['Outcome'].fillna(-1).astype('int')
    ddict['group'] = df[['AgeGroup', 'SubjNum']].drop_duplicates()['AgeGroup']

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

    with pd.ExcelWriter(outfile) as writer:
        for sub in xrange(ddict['Nsub']):
            print "Writing subject {}".format(sub)
            df = pd.DataFrame(D[sub])
            df.to_excel(writer, sheet_name='RPE_Subject' + str(sub))
            df = pd.DataFrame(Q[sub])
            df.to_excel(writer, sheet_name='EV_Subject' + str(sub))

        df = pd.DataFrame(sub_alpha)
        df.to_excel(writer, sheet_name='Learning Rates')

    try:
        # write out predicted learning rate samples for plotting
        if len(samples['alpha_pred'].shape) == 1:
            cols = ['Younger']
        else:
            cols = ['Younger', 'Older']
        preds = pd.DataFrame(samples['alpha_pred'], columns=cols)
        preds.to_csv('Model_preds.csv')
    except:
        pass