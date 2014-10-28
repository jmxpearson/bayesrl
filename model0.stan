data {
    int<lower = 0> N;  // number of observations
    int<lower = 0> Nsub;  // number of subjects
    int<lower = 0> Ncue;  // number of cues
    int<lower = 0> Ntrial;  // number of trials per subject
    int<lower = 0> sub[N];  // subject index 
    int<lower = 0> chosen[N];  // index of chosen option
    int<lower = 0> unchosen[N];  // index of unchosen option 
    int<lower = 1> trial[N];  // trial number
    int<lower = -1, upper = 1> outcome[N];  // outcome: -1 => missing
}

parameters {
    vector[Nsub] beta;  // softmax parameter
    real<lower = 0, upper = 1> alpha[Nsub];  // learning rate
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
        else {  // carry forward last trial's values
            for (c in 1:Ncue) {
                Q[sub[idx], trial[idx], c] <- Q[sub[idx], trial[idx] - 1, c];
                Delta[sub[idx], trial[idx], c] <- 0;
            }
        }

        if (outcome[idx] >= 0) {
                Delta[sub[idx], trial[idx], chosen[idx]] <- outcome[idx] - Q[sub[idx], trial[idx], chosen[idx]];

                Q[sub[idx], trial[idx], chosen[idx]] <- Q[sub[idx], trial[idx], chosen[idx]] + alpha[sub[idx]] * Delta[sub[idx], trial[idx], chosen[idx]];
                //print("chosen = ", chosen[idx], " unchosen = ", unchosen[idx]);
        }
    }
}
 
model {
    beta ~ normal(0, 5);
    alpha ~ beta(1, 1);
    for (idx in 1:N) {
        if (chosen[idx] > 0) {
            1 ~ bernoulli_logit(beta[sub[idx]] * (Q[sub[idx], trial[idx], chosen[idx]] - Q[sub[idx], trial[idx], unchosen[idx]]));
        }
    }
}
