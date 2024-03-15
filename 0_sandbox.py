"""Simple sandbox."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>
#          Hugo Richard <research.hugo.richard@gmail.com>

import os
import time
import copy
import matplotlib.pyplot as plt

import numpy as np
from bandpy import utils, runners

from toolbox import MAX_RANDINT, IndependantController, GaussianKBanditPlayers, UCB, EpsGreedy


###############################################################################
# Globals
###############################################################################

print(f"[Main] File: '{os.path.basename(__file__)}'")

t0_total = time.time()

fontsize = 15
seed = None
n_trials = 250
trial_to_plot = [44, 14, 5, 16, 1]
n_jobs = -2
verbose = False
rng = utils.check_random_state(seed)
dpi = 650

plot_dir = "_figures_async_players"
if not os.path.isdir(plot_dir):
    os.makedirs(plot_dir)

###############################################################################
# Experiment setting
###############################################################################
T = 500
K = 2
N = 2

# mu_perturbation = np.rng.normal(K)
mu_perturbation = np.linspace(-0.5, 0.5, K)
# mu_perturbation = np.array([0.0] * int((K+1)/2) + [-1.0] * int(K/2))

sigma_arm = 0.5
mu_offset = 1.0
mu = mu_offset + sigma_arm * mu_perturbation  # each mu should be higher than 0 with high prob.
mu_min = np.sort([np.max(mu) - mu_i for mu_i in mu])[1]
sigma = 1.0
delta = 0.1
R_T_max = T * (np.max(mu) - np.min(mu))
ylim_R_t = T / 2

agent_cls = UCB
agent_kwargs = {"delta": delta, "K": len(mu), "seed": seed}
# agent_cls = EpsGreedy
# agent_kwargs = {"K": len(mu), "seed": seed, "eps": 0.1}

###############################################################################
# Main
###############################################################################
print(f"[Main] Main experiment running:")
print(f"[Main] {K}-arms: " + ' '.join([f"{mu_k:.3f}" for mu_k in mu]))

# l_p = np.linspace(0.1, 1.0, 10)
l_p = (1.0 - np.logspace(-3, np.log10(0.9), 5))[::-1]

l_p_to_plot_evolution = l_p

all_results = dict()
for p in l_p:

    print(f"[Main] Running p={p:.4f} ...", end=' ')

    t0_run = time.time()

    bandit_env = GaussianKBanditPlayers(N=N, mu=mu, sigma=sigma, T=T, seed=seed)
    bandit_controller = IndependantController(N=N, p=p, agent_cls=agent_cls,
                                              agent_kwargs=agent_kwargs)

    seeds = rng.integers(MAX_RANDINT, size=n_trials)

    results = runners.run_trials(copy.deepcopy(bandit_env),
                                 copy.deepcopy(bandit_controller),
                                 early_stopping=False, seeds=seeds,
                                 n_jobs=n_jobs, verbose=verbose)

    print(f"done in {time.time() - t0_run:.1f} s")

    all_results[p] = results

###########################################################################
# Plotting
###########################################################################

## R_T histogram

all_fig, all_axis = dict(), dict()
for p in l_p:

    results = all_results[p]
    all_R_T_per_p = [np.mean([np.sum(env.r_t[n][env.r_t[n] != -np.inf]) for n in range(N)]) for _, env in results]

    fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 2.25), squeeze=False, sharey=True)

    _, _, bins = plt.hist(all_R_T_per_p, bins=5)
    axis[0, 0].bar_label(bins, color='tab:blue', alpha=0.5, fontsize=5)
    axis[0, 0].axvline(R_T_max, color='tab:gray', lw=1.5, alpha=0.2)

    axis[0, 0].set_xlabel(r'$R_T$', fontsize=int(0.75*fontsize))
    axis[0, 0].set_ylabel('Count', rotation=90, fontsize=int(0.8*fontsize))

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
    title = f"N={N} K={len(mu)} p={p:.4f} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}"
    all_axis[p][0, 0].text(-0.1*x_lim_max, 1.25*y_lim_max, title, ha='left', va='center', fontsize=int(0.75*fontsize))

    all_fig[p].tight_layout()

    for filename in [f'hist_R_T__p_{p:.4f}.pdf', f'hist_R_T__p_{p:.4f}.png']:
        filepath = os.path.join(plot_dir, filename)
        all_fig[p].savefig(filepath, dpi=dpi)
        print(f"[Main] Saving {filepath}")

## s_t evolution

for p in l_p_to_plot_evolution:

    results = all_results[p]

    ncols = len(trial_to_plot)
    nrows = N
    fig, axis = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2.5, nrows * 2.0),
                             squeeze=False, sharex=True, sharey=True, tight_layout=True)

    for i, j in enumerate(trial_to_plot):

        controller = results[j][0]
        env = results[j][1]

        for n in range(N):

            agent_name = f'agent_{n}'

            tt_collision = np.array(list(env.t_collision.keys()), dtype=int)

            # visual ref for arms
            for mu_k in mu:
                axis[n, i].axhline(mu_k, color='tab:gray', lw=1.0, alpha=0.2)

            # visual ref for collision
            for t_c in tt_collision:
                axis[n, i].axvline(t_c, color='tab:orange', lw=1.0, alpha=0.1)

            # instantaneous pseudo-regret
            tt_awake = np.array(controller.agents[agent_name].tt_awake) + 2
            no_noise_s_t = np.array(env.no_noise_s_t[n])[tt_awake-1]

            axis[n, i].plot(tt_awake, no_noise_s_t, linestyle='', marker='o',
                            color='tab:blue', markersize=2.5, alpha=0.2)

            # instantaneous pseudo-regret during a collision (if the pseudo-regret was not set to 0)
            tt_c_awake = np.array([t_c_awake for t_c_awake in tt_awake if t_c_awake in tt_collision])
            no_noise_s_t_collision = []
            for t_c_awake in tt_c_awake:
                    k = env.t_collision[t_c_awake][agent_name]
                    no_noise_s_t_collision.append(mu[k])

            axis[n, i].plot(tt_c_awake, no_noise_s_t_collision, linestyle='', marker='*',
                            color='tab:orange', markersize=3.5, alpha=0.35)

            # set y limits
            axis[n, i].set_ylim(0.0, 1.1 * np.max(mu))

    for i, j in enumerate(trial_to_plot):
        n_collisions = len(list(results[j][1].t_collision.keys()))
        axis[N-1, i].set_xlabel(f'trial-{j}\n({n_collisions} collisions)', fontsize=fontsize)

    for n in range(N):
        axis[n, 0].set_ylabel(f'agent-{n+1}', fontsize=fontsize)

    title = f"N={N} K={len(mu)} p={p:.4f} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}"
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()

    for filename in [f'evolution_s_t__p_{p:.4f}.pdf', f'evolution_s_t__p_{p:.4f}.png']:
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=dpi)
        print(f"[Main] Saving {filepath}")

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

            valid_mask = all_r_t[:, t] != -np.inf
            valid_r_t = all_r_t[valid_mask, t]

            if valid_r_t.size != 0:
                mean_r_t = np.mean(valid_r_t)
            else: mean_r_t = np.nan

            if not np.isnan(mean_r_t):
                mean_R_t[t] = mean_R_t[t-1] + mean_r_t
            else: mean_R_t[t] = mean_R_t[t-1]

        axis[0, i].plot(mean_R_t, color='tab:blue', lw=3.5, alpha=0.5)
        axis[0, i].plot(np.arange(T), color='tab:gray', lw=2.5, alpha=0.5)
        axis[0, i].set_ylim(0.0, ylim_R_t)

        for t_c in env.t_collision:
            axis[0, i].axvline(t_c, color='tab:gray', lw=1.5, alpha=0.1)

    for i, j in enumerate(trial_to_plot):
        n_collisions = len(list(results[j][1].t_collision.keys()))
        axis[0, i].set_xlabel(f'trial-{j}\n({n_collisions} collisions)', fontsize=fontsize)

    title = f"N={N} K={len(mu)} p={p:.4f} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}"
    fig.suptitle(title, fontsize=fontsize)
    fig.tight_layout()

    for filename in [f'evolution_R_t__p_{p:.4f}.pdf', f'evolution_R_t__p_{p:.4f}.png']:
        filepath = os.path.join(plot_dir, filename)
        fig.savefig(filepath, dpi=dpi)
        print(f"[Main] Saving {filepath}")

## Nb collision evolution wrt p

mean_n_collisions, std_n_collisions = [], []
for p in l_p_to_plot_evolution:

    all_n_collisions_per_p = [len(list(all_results[p][i][1].t_collision.keys())) for i in range(n_trials)]

    mean_n_collisions.append(np.mean(all_n_collisions_per_p))
    std_n_collisions.append(np.std(all_n_collisions_per_p))

mean_n_collisions = np.array(mean_n_collisions)
std_n_collisions = np.array(std_n_collisions)

fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 2.5), squeeze=False, sharex=True, sharey=True, tight_layout=True)

axis[0, 0].plot(l_p, mean_n_collisions, color='tab:blue', lw=3.5, alpha=0.5)
axis[0, 0].fill_between(l_p, mean_n_collisions - std_n_collisions,
                        mean_n_collisions + std_n_collisions, color='tab:blue', alpha=0.25)

axis[0, 0].set_xlabel("p", fontsize=fontsize)
axis[0, 0].set_ylabel("Nb collision", fontsize=fontsize)
axis[0, 0].set_title(f"N={N} K={len(mu)} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}", fontsize=fontsize)

fig.tight_layout()

for filename in [f'evolution_n_collisions_wrt_p.pdf', f'evolution_n_collisions_wrt_p.png']:
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath, dpi=dpi)
    print(f"[Main] Saving {filepath}")

## R_T evolution wrt p

mean_R_T, std_R_T = [], []
for p in l_p_to_plot_evolution:

    results = all_results[p]

    all_R_T_per_p = [np.mean([np.sum(env.r_t[n][env.r_t[n] != -np.inf]) for n in range(N)]) for _, env in results]

    mean_R_T.append(np.mean(all_R_T_per_p))
    std_R_T.append(np.std(all_R_T_per_p))

mean_R_T = np.array(mean_R_T)
std_R_T = np.array(std_R_T)

fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(3.0, 2.5), squeeze=False, sharex=True, sharey=True, tight_layout=True)

axis[0, 0].plot(l_p, mean_R_T, color='tab:blue', lw=3.5, alpha=0.5)
axis[0, 0].fill_between(l_p, mean_R_T - std_R_T, mean_R_T + std_R_T, color='tab:blue', alpha=0.25)

axis[0, 0].set_xlabel("p", fontsize=fontsize)
axis[0, 0].set_ylabel(r'$R_T$', fontsize=fontsize)
axis[0, 0].set_title(f"N={N} K={len(mu)} " + r"$\Delta_{\mu}^{\min}$= " + f"{mu_min:.2f}", fontsize=fontsize)

fig.tight_layout()

for filename in [f'evolution_R_T_wrt_p.pdf', f'evolution_R_T_wrt_p.png']:
    filepath = os.path.join(plot_dir, filename)
    fig.savefig(filepath, dpi=dpi)
    print(f"[Main] Saving {filepath}")

###############################################################################
# Runtime
###############################################################################

delta_t = time.gmtime(time.time() - t0_total)
delta_t = time.strftime("%H h %M min %S s", delta_t)
print(f"[Main] Script runs in {delta_t}")
