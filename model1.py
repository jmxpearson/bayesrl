"""
Runs Model1, a Bayesian RL model with hierarchical group effects, on 
input csv file. Usage:
python model1.py <infile> -o <outfile>
If <outfile> is unspecified, defaults to "results.xlsx"
"""
from __future__ import division
import numpy as np
import pandas as pd
import pystan
import sys

if __name__ == '__main__':

    # file names
    infile = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == '-o':
        outfile = sys.argv[3]
    else:
        outfile = "results.xlsx"

    # read in
    df = pd.read_csv(infile)

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

    # compile stan model
    np.random.seed(77752)
    sm = pystan.StanModel(file='model1.stan')

    # choose inits
    def initfun():
        return {'beta': np.ones((ddict['Nsub'],)), 
                'alpha': np.random.rand(ddict['Nsub']),
                'a': np.ones((ddict['Ngroup'],)),
                'b': np.ones((ddict['Ngroup'],))}

    # run it
    fit = sm.sampling(data=ddict, chains=2, init=initfun)

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