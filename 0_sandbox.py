"""Simple sandbox."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>
# Authors: Hugo Richard <research.hugo.richard@gmail.com>

import os
import time
import copy
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from bandpy import utils, runners

from toolbox import IndependantController, GaussianKBanditPlayers, UCB


###############################################################################
# Globals
###############################################################################

print(f"[Main] Experiment: {os.path.basename(__file__)}.")

t0_total = time.time()

fontsize = 18
MAX_RANDINT = 1000
seed = None
n_trials = 5
n_jobs = 5
verbose = False
rng = utils.check_random_state(seed)
agent_cls = UCB
dpi = 300
sns.set_theme()
colors = ['tab:blue', 'tab:orange', 'tab:green']

plot_dir = "figures"
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

###############################################################################
# Experiment setting
###############################################################################
T = 250
mu_offset = 1.0
mu = mu_offset + np.array([0.5, 0.1, 0.3, 0.55])  # should be higher than 0
N = 3
assert N <= len(mu), "The number of agents should not exceed than the number of arms"
assert N == len(colors), "The number of colors should match the number of agents for the figures"
sigma = 1.0

###############################################################################
# Main
###############################################################################
print(f"[Main] Main experiment running:")

for p in np.linspace(0.1, 1.0, 5):

    print(f"[Main] Running p={p} ...")

    agent_kwargs = {"delta": 0.1, "K": len(mu), "seed": seed}

    bandit_env = GaussianKBanditPlayers(N=N, mu=mu, sigma=sigma, T=T, seed=seed)
    bandit_controller = IndependantController(N=N, p=p, agent_cls=agent_cls,
                                            agent_kwargs=agent_kwargs)

    seeds = rng.randint(MAX_RANDINT, size=n_trials)

    results = runners.run_trials(copy.deepcopy(bandit_env),
                                 copy.deepcopy(bandit_controller),
                                 early_stopping=False, seeds=seeds,
                                 n_jobs=n_jobs, verbose=verbose)

    ###########################################################################
    # Plotting
    ###########################################################################

    ## Swarmplot

    all_R_T = pd.DataFrame(columns=['agent_id', 'R_T'])
    for _, env in results:
        for n in range(N):
            R_T = np.sum(env.r_t[n][env.r_t[n] != -np.inf])
            all_R_T.loc[len(all_R_T)] = [n, R_T]

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 2.0), squeeze=False,
                             sharey=True, tight_layout=True)

    sns.violinplot(data=all_R_T, x='agent_id', y='R_T', ax=axis[0, 0], alpha=0.5)

    axis[0, 0].set_xticklabels(range(1, N+1))
    axis[0, 0].set_xlabel('Agent ID', fontsize=int(0.75*fontsize))
    axis[0, 0].set_ylabel(r'$R_T$', rotation=90, fontsize=int(0.75*fontsize))
    axis[0, 0].set_title(r'$p=' + f'{p}' + '$', fontsize=fontsize)

    for filename in [f'swarm_R_T__p_{p}.pdf', f'swarm_R_T__p_{p}.png']:
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=dpi)
        print(f"[Main]   Saving {filepath}.")

    ## Wake-up ratio

    T_i = []
    for i in range(n_trials):
        controller = results[i][0]
        T_i.append(controller.T_i / np.sum(controller.T_i))

    median_T_i = utils.tolerant_median(T_i)
    std_T_i = utils.tolerant_std(T_i)

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(2.0, 2.0),
                            squeeze=False, tight_layout=True)

    axis[0, 0].bar(x=range(1, N+1), height=median_T_i, yerr=std_T_i, color=colors)

    fig.tight_layout()

    for filename in [f'wake_up_ratio__p_{p}.pdf', f'wake_up_ratio__p_{p}.png']:
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=dpi)
        print(f"[Main]   Saving {filepath}.")

    ## R_t evolution

    ncols = n_trials
    nrows = N
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 7.0, nrows * 2.0),
                             squeeze=False, sharex=True, sharey=True, tight_layout=True)

    for i in range(n_trials):
        env = results[i][1]
        for n in range(N):
            axis[n, i].plot(env.no_noise_s_t[n], c=colors[n], lw=3.0, alpha=0.5)
            axis[n, i].set_ylim(0.0, np.max(mu))
            for t_c in env.t_collision:
                axis[n, i].axvline(t_c, c='black', lw=1.5, alpha=0.2)

    for i in range(n_trials):
        axis[N-1, i].set_xlabel(f'trial-{i}', fontsize=fontsize)

    fig.text(0.0, 0.5, 'Agent', va='center', rotation='vertical', fontsize=fontsize)

    fig.tight_layout()

    for filename in [f'evolution_s_t__p_{p}.pdf', f'evolution_s_t__p_{p}.png']:
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=dpi)
        print(f"[Main]   Saving {filepath}.")

###############################################################################
# Runtime
###############################################################################

delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
print(f"[Main] Script runs in {delta_t}")
