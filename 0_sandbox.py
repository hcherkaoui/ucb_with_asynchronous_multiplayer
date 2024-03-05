"""Simple sandbox."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>
# Authors: Hugo Richard <research.hugo.richard@gmail.com>

import os
import time
import copy
import matplotlib.pyplot as plt

import numpy as np
from bandpy import utils, runners

from toolbox import IndependantController, GaussianKBanditPlayers, UCB


###############################################################################
# Globals
###############################################################################

print(f"[Main] Experiment: {os.path.basename(__file__)}.")

t0_total = time.time()

fontsize = 17
MAX_RANDINT = 1000
seed = 0
n_trials = 100
trial_to_plot = [44, 14, 5]
n_jobs = -2
verbose = False
rng = utils.check_random_state(seed)
agent_cls = UCB
dpi = 200

plot_dir = "_figures_async_players"
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

###############################################################################
# Experiment setting
###############################################################################
T = 1000
K = 20
N = 5
sigma_arm = 0.2
mu_offset = 2.0
mu = mu_offset + sigma_arm * rng.randn(K)  # each mu should be higher than 0 with high prob.
mu_min = np.sort([np.max(mu) - mu_i for mu_i in mu])[1]
sigma = 1.0
delta = 0.1
R_T_max = T * (np.max(mu) - np.min(mu))

###############################################################################
# Main
###############################################################################
print(f"[Main] Main experiment running:")

l_p = np.linspace(0.1, 1.0, 6)
l_p_to_plot_evolution = l_p
all_results = dict()
for p in l_p:

    print(f"[Main] Running p={p:.3f} ...", end=' ')

    agent_kwargs = {"delta": delta, "K": len(mu), "seed": seed}

    bandit_env = GaussianKBanditPlayers(N=N, mu=mu, sigma=sigma, T=T, seed=seed)
    bandit_controller = IndependantController(N=N, p=p, agent_cls=agent_cls,
                                              agent_kwargs=agent_kwargs)

    seeds = rng.randint(MAX_RANDINT, size=n_trials)

    results = runners.run_trials(copy.deepcopy(bandit_env),
                                 copy.deepcopy(bandit_controller),
                                 early_stopping=False, seeds=seeds,
                                 n_jobs=n_jobs, verbose=verbose)

    print("done")

    all_results[p] = results

###########################################################################
# Plotting
###########################################################################

## R_T histogram

all_fig, all_axis = dict(), dict()
for p in l_p:

    results = all_results[p]
    all_R_T = [np.sum(env.r_t[n][env.r_t[n] != -np.inf]) for n in range(N) for _, env in results]

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 2.5), squeeze=False, sharey=True)

    _, _, bins = plt.hist(all_R_T, bins=5)
    axis[0, 0].bar_label(bins, color='tab:blue', alpha=0.5)
    axis[0, 0].axvline(R_T_max, c='tab:gray', lw=1.5, alpha=0.2)

    axis[0, 0].set_xlabel(r'$R_T$', fontsize=int(0.75*fontsize))
    axis[0, 0].set_ylabel('Count', rotation=90, fontsize=int(0.75*fontsize))
    title = f"N={N} K={len(mu)} p={p:.2f} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}"
    axis[0, 0].set_title(title, fontsize=fontsize)

    all_fig[p] = fig
    all_axis[p] = axis

l_x_lim, l_y_lim = [], []
for p in l_p:

    _, x_lim = all_axis[p][0, 0].get_xlim()
    _, y_lim = all_axis[p][0, 0].get_ylim()

    l_x_lim.append(x_lim)
    l_y_lim.append(y_lim)

x_lim_max = np.max(l_x_lim)
y_lim_max = np.max(l_y_lim)

for p in l_p:

    all_axis[p][0, 0].set_xlim(0.0, 1.1 * x_lim_max)
    all_axis[p][0, 0].set_ylim(0.0, 1.1 * y_lim_max)
    all_axis[p][0, 0].text(R_T_max, 0.25 * y_lim_max, "$R_T^{\mathrm{max}}="+f"{R_T_max:.1f}"+"$",
                           color='tab:gray', rotation=90, fontsize=int(0.4*fontsize))

    all_fig[p].tight_layout()

    for filename in [f'hist_R_T__p_{p:.3f}.pdf', f'hist_R_T__p_{p:.3f}.png']:
        filepath = os.path.join(plot_dir, filename)
        all_fig[p].savefig(filepath, dpi=dpi)
        print(f"[Main]   Saving {filepath}.")

## s_t evolution

for p in l_p_to_plot_evolution:

    results = all_results[p]

    ncols = len(trial_to_plot)
    nrows = N
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.5, nrows * 2.0),
                             squeeze=False, sharex=True, sharey=True, tight_layout=True)

    for i, j in enumerate(trial_to_plot):

        env = results[j][1]

        for n in range(N):

            axis[n, i].plot(env.no_noise_s_t[n], linestyle='', marker='o', c='tab:blue',
                            lw=3.0, alpha=0.5)
            axis[n, i].set_ylim(0.0, np.max(mu))

            for t_c in env.t_collision:
                axis[n, i].axvline(t_c, c='black', lw=1.5, alpha=0.2)

    for i, j in enumerate(trial_to_plot):
        axis[N-1, i].set_xlabel(f'trial-{j}', fontsize=fontsize)

    for n in range(N):
        axis[n, 0].set_ylabel(f'agent-{n}', fontsize=fontsize)

    title = f"N={N} K={len(mu)} p={p:.2f} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}"
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()

    for filename in [f'evolution_s_t__p_{p:.3f}.pdf', f'evolution_s_t__p_{p:.3f}.png']:
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=dpi)
        print(f"[Main]   Saving {filepath}.")

## R_t evolution

for p in l_p_to_plot_evolution:

    results = all_results[p]

    ncols = len(trial_to_plot)
    nrows = 1
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.5, 2.0),
                             squeeze=False, sharex=True, sharey=True, tight_layout=True)

    for i, j in enumerate(trial_to_plot):

        env = results[j][1]

        all_r_t = []
        for n in range(N):
            all_r_t.append(env.no_noise_r_t[n])
        all_r_t = np.array(all_r_t)

        mean_R_t = np.zeros((T,))
        for t in range(1, T):

            mean_r_t = np.mean(all_r_t[:, t][all_r_t[:, t] != -np.inf])

            if not np.isnan(mean_r_t):
                mean_R_t[t] = mean_R_t[t-1] + mean_r_t

            else: mean_R_t[t] = mean_R_t[t-1]

        axis[0, i].plot(mean_R_t, c='tab:blue', lw=3.5, alpha=0.5)
        axis[0, i].plot(np.arange(T), c='tab:gray', lw=2.5, alpha=0.5)

        for t_c in env.t_collision:
            axis[0, i].axvline(t_c, c='black', lw=1.5, alpha=0.2)

    for i, j in enumerate(trial_to_plot):
        axis[0, i].set_xlabel(f'trial-{j}', fontsize=fontsize)

    title = f"N={N} K={len(mu)} p={p:.2f} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}"
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()

    for filename in [f'evolution_R_t__p_{p:.3f}.pdf', f'evolution_R_t__p_{p:.3f}.png']:
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=dpi)
        print(f"[Main]   Saving {filepath}.")

###############################################################################
# Runtime
###############################################################################

delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
print(f"[Main] Script runs in {delta_t}")
