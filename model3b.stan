// Model 3:
// Model 3, but only updates chosen option

data {
    int<lower = 1> N;  // number of observations
    int<lower = 1> Nsub;  // number of subjects
    int<lower = 1> Ncue;  // number of cues
    int<lower = 1> Ntrial;  // number of trials per subject
    int<lower = 1> Ngroup;  // number of experimental groups 
    int<lower = 1> Ncond;  // number of task conditions
    int<lower = 1> Nrun;  // number of task conditions
    int<lower = 0> sub[N];  // subject index 
    int<lower = 0> chosen[N];  // index of chosen option: 0 => missing
    int<lower = 0> unchosen[N];  // index of unchosen option: 0 => missing
    int<lower = 0> condition[N];  // delay condition code: 0 => missing
    int<lower = 0> run[N];  // run number code: 0 => missing
    int<lower = 1> trial[N];  // trial number
    int<lower = -1, upper = 1> outcome[N];  // outcome: -1 => missing
    int<lower = 1> group[Nsub];  // group assignment for each subject
}

parameters {
    vector<lower = 0>[Nsub] beta;  // softmax parameter
    real<lower = 0, upper = 1> alpha[Nsub, Nrun];  // learning rate
    real<lower = 0> a[Ngroup, Nrun];  // parameter for group-specific alpha
    real<lower = 0> b[Ngroup, Nrun];  // parameter for group-specific alpha
}

transformed parameters {
    real<lower=0, upper=1> Q[Nsub, Ntrial, Ncue];  // value function for each target
    real Delta[Nsub, Ntrial, Ncue];  // prediction error

    for (idx in 1:N) {
        if (trial[idx] == 1) {
            for (c in 1:Ncue) {
                Q[sub[idx], trial[idx], c] <- 0.5;
                Delta[sub[idx], trial[idx], c] <- 0;
            }
        }
        if (trial[idx] < Ntrial) {  // push forward this trial's values
            for (c in 1:Ncue) {
                Q[sub[idx], trial[idx] + 1, c] <- Q[sub[idx], trial[idx], c];
                Delta[sub[idx], trial[idx], c] <- 0;
            }
        }

        if (outcome[idx] >= 0) {
                // prediction error: chosen option
                Delta[sub[idx], trial[idx], chosen[idx]] <- outcome[idx] - Q[sub[idx], trial[idx], chosen[idx]];

                if (trial[idx] < Ntrial) {  // update action values for next trial
                    // update chosen option
                    Q[sub[idx], trial[idx] + 1, chosen[idx]] <- Q[sub[idx], trial[idx], chosen[idx]] + alpha[sub[idx], run[idx]] * Delta[sub[idx], trial[idx], chosen[idx]];
                }
        }
    }
}

model {
    beta ~ gamma(1, 0.2);
    for (grp in 1:Ngroup) {
        for (rnum in 1:Nrun) {
            a[grp, rnum] ~ gamma(1, 1);
            b[grp, rnum] ~ gamma(1, 1);
        }
    }

    for (idx in 1:Nsub) {
        for (rnum in 1:Nrun) {
            alpha[idx, rnum] ~ beta(a[group[idx], rnum], b[group[idx], rnum]);
        }
    }

    for (idx in 1:N) {
        if (chosen[idx] > 0) {
            1 ~ bernoulli_logit(beta[sub[idx]] * (Q[sub[idx], trial[idx], chosen[idx]] - Q[sub[idx], trial[idx], unchosen[idx]]));
        }
    }
}

generated quantities {  // generate samples of learning rate from each group
    real<lower=0, upper=1> alpha_pred[Ngroup, Nrun];
    for (grp in 1:Ngroup) {
        for (rnum in 1:Nrun) {
            alpha_pred[grp, rnum] <- beta_rng(a[grp, rnum], b[grp, rnum]);

        }
    }
}
 
