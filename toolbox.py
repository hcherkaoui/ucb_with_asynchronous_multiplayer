"""Toolbox module."""

# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>
#          Hugo Richard <research.hugo.richard@gmail.com>

import numbers
import numpy as np
from bandpy import _checks


MAX_RANDINT = 1000


class GaussianKBanditPlayers:

    def __init__(self, N, mu, sigma, T, seed=None):
        """Init."""
        self.T = T
        self.t = 1

        self.N = N

        self.mu = mu
        if isinstance(sigma, numbers.Number):
            self.sigma = [sigma] * len(self.mu)
        else: self.sigma = sigma

        self.seed = seed
        self.rng = _checks.check_random_state(self.seed)

        self.best_arm = np.argmax(self.mu)
        self.best_reward = np.max(self.mu)

        self.init_metrics()

        self.t_collision = dict()

    def init_metrics(self):
        """Init/reset all the statistics."""
        default_value = -np.inf
        self.s_t = {i: default_value * np.ones((self.T,)) for i in range(self.N)}
        self.no_noise_s_t = {i: default_value * np.ones((self.T,)) for i in range(self.N)}
        self.best_s_t = {i: default_value * np.ones((self.T,)) for i in range(self.N)}
        self.worst_s_t = {i: default_value * np.ones((self.T,)) for i in range(self.N)}
        self.r_t = {i: default_value * np.ones((self.T,)) for i in range(self.N)}
        self.no_noise_r_t = {i: default_value * np.ones((self.T,)) for i in range(self.N)}

    def reset(self, seed=None):
        """Reset internal statistics."""
        self.seed = seed
        self.rng = _checks.check_random_state(self.seed)
        self.init_metrics()
        self.t = 1

    def compute_reward(self, actions):
        """Compute the reward, return 0 if collision occured."""
        all_k = list(actions.values())

        N_awake = len(all_k)

        _, c = np.unique(all_k, return_counts=True)

        if any(c > 1):
            self.t_collision[self.t] = actions
            return np.array([0.0] * N_awake), np.array([0.0] * N_awake)

        else:
            noise = np.array([self.sigma[k] * self.rng.normal() for k in all_k])
            no_noise_y = np.array([self.mu[k] for k in all_k])
            return no_noise_y + noise, no_noise_y

    def update_agent_stats(self, agent_names, all_y, all_no_noise_y):
        """Update internal statistics for a given agent."""
        y_max, y_min = np.max(self.mu), np.min(self.mu)
        for agent_name, y, no_noise_y in zip(agent_names, all_y, all_no_noise_y):
            self._update_agent_stats(agent_name, y, no_noise_y, y_max, y_min)

    def _update_agent_stats(self, agent_name, y, no_noise_y, y_max, y_min):
        """Update all statistic as listed in __init__ doc."""
        i = int(agent_name.split('_')[1])
        self.s_t[i][self.t - 1] = y
        self.no_noise_s_t[i][self.t - 1] = no_noise_y
        self.best_s_t[i][self.t - 1] = y_max
        self.worst_s_t[i][self.t - 1] = y_min
        self.r_t[i][self.t - 1] = y_max - y
        self.no_noise_r_t[i][self.t - 1] = y_max - no_noise_y

    def step(self, action):
        """Pull the k-th arm chosen in 'actions'."""
        action = _checks.check_actions(action)

        all_y, all_no_noise_y = self.compute_reward(action)

        self.update_agent_stats(action.keys(), all_y, all_no_noise_y)

        info, observation = dict(), dict()
        for agent_name, k, y, no_noise_y in zip(action.keys(), action.values(), all_y, all_no_noise_y):
            observation[agent_name] = {"last_arm_pulled": k,
                                       "last_reward": y,
                                       "last_no_noise_reward": no_noise_y,
                                       "t": self.t,
                                       }
            info[agent_name] = {"best_arm": self.best_arm,
                                "best_reward": self.best_reward,
                                "seed": self.seed
                                }

        reward = {agent_name: y for y, agent_name in zip(all_y, action.keys())}

        self.t += 1

        done = self.T < self.t

        return observation, reward, done, info


class IndependantController:

    def __init__(self,
                 p,
                 agent_cls,
                 agent_kwargs,
                 N=None,
                 agent_names=None,
                 seed=None,
                 ):
        """Init."""
        N, agent_names = _checks.check_N_and_agent_names(N, agent_names)

        self.p = p

        self.counter = 0  # XXX

        self.agents = dict()

        self.N = N

        self.t = -1  # to be synchronous with the env count
        self.T_i = np.zeros((self.N,), dtype=int)
        self.t_i = {i: [] for i in range(self.N)}

        self.seed = seed
        self.rng = _checks.check_random_state(seed)

        if agent_names is None:
            self.agent_names = [f"agent_{i}" for i in range(self.N)]
        else:
            self.agent_names = agent_names

        self.agent_cls = agent_cls
        self.agent_kwargs = agent_kwargs

        for agent_name in self.agent_names:
            self.agent_kwargs['seed'] = self.rng.integers(MAX_RANDINT)
            self.agents[agent_name] = self.agent_cls(**self.agent_kwargs)

        self.done = False

    def reset(self, seed=None):
        """Reset internal statistics."""

        self.seed = seed
        self.rng = _checks.check_random_state(self.seed)

        for agent_name in self.agent_names:
            self.agent_kwargs['seed'] = self.rng.integers(MAX_RANDINT)
            self.agents[agent_name] = self.agent_cls(**self.agent_kwargs)

        self.t = -1
        self.T_i = np.zeros((self.N,), dtype=int)
        self.t_i = {i: [] for i in range(self.N)}

    @property
    def best_arms(self):
        """Return for each agent the estimated best arm."""
        best_arms = dict()
        for agent_name, agent in self.agents.items():
            best_arms[agent_name] = agent.best_arm
        return best_arms

    def default_act(self):
        """Choose one agent and makes it pull the 'default' arm to init the
        simulation."""
        agent_names = self._choose_agent()
        return {agent_name: self.agents[agent_name].select_default_arm()
                for agent_name in agent_names}

    def _check_if_controller_done(self):
        """Check if the controller return 'done'."""
        dones = [agent.done for agent in self.agents.values() if hasattr(agent, "done")]  # noqa
        self.done = (len(dones) != 0) & all(dones)

    def _choose_agent(self):
        awake_agent_i = [i for i in range(self.N) if self.rng.uniform() < self.p]
        self.T_i[awake_agent_i] += 1
        for i in awake_agent_i:
            self.t_i[i].append(self.t)
        return [f"agent_{i}" for i in awake_agent_i]

    def act(self, observation, reward, info):
        """Make each agent choose an arm in a clustered way."""

        self.t += 1

        # update agent stats
        for last_agent_name, agent_observation in observation.items():
            last_k = agent_observation["last_arm_pulled"]
            last_r = agent_observation["last_reward"]
            self.agents[last_agent_name]._update_local(last_k, last_r)

        # trigger action
        action = dict()
        agent_names = self._choose_agent()
        for agent_name in agent_names:
            selected_k = self.agents[agent_name].act(self.t)
            action[agent_name] = selected_k

        # check if done
        self._check_if_controller_done()

        return action


class UCB:
    """Upper confidence bound class to define the UCB algorithm. """

    def __init__(self, K, delta, T_0=None, random_break_tie=False, seed=None):
        """Init."""

        self.K = K
        self.T_0 = T_0
        self.delta = delta

        self.rng = _checks.check_random_state(seed)
        self.random_break_tie = random_break_tie

        self.n_pulls_per_arms = dict([(k, 0) for k in range(self.K)])
        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])
        self.tt_awake = []

    def select_default_arm(self):
        """Select the 'default arm'."""
        self.tt_awake.append(-1)  # to be synchronous with the env count
        k = self.rng.integers(self.K)
        return k

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k]) for k in range(self.K)]
        return np.argmax(mean_reward_per_arms)

    def _update_local(self, last_k, last_r):
        """Update local variables."""
        self.n_pulls_per_arms[last_k] += 1
        self.reward_per_arms[last_k].append(last_r)

    def act(self, t):
        """Select an arm."""
        self.tt_awake.append(t)

        if self.T_0 is not None and t < self.T_0:
            k = self.rng.integers(self.K)

        else:
            uu = []
            for k in range(self.K):

                T_k = self.n_pulls_per_arms[k]
                mu_k = np.mean(self.reward_per_arms[k])

                if T_k == 0:
                    uu.append(np.inf)
                else:
                    uu.append(mu_k + np.sqrt(2 * np.log(1.0 / self.delta) / T_k))

            if self.random_break_tie:
                k = self.rng.choice(np.argwhere(uu == np.amax(uu)).flatten())  # if there is a tie draw randomly
            else: k = np.argmax(uu)

        return k


class EpsGreedy:
    """EpsGreedy class to define a simple agent that choose the best arm
    observed so far."""

    def __init__(self, K, eps=0.0, T_0=None, random_break_tie=False, seed=None):
        """Init."""

        self.K = K
        self.eps = eps
        self.T_0 = T_0

        self.rng = _checks.check_random_state(seed)
        self.random_break_tie = random_break_tie

        self.reward_per_arms = dict([(k, [0.0]) for k in range(self.K)])
        self.tt_awake = []

    def select_default_arm(self):
        """Select the 'default arm'."""
        self.tt_awake.append(-1)  # to be synchronous with the env count
        k = self.rng.integers(self.K)
        return k

    @property
    def best_arm(self):
        """Return the estimated best arm if the estimation is avalaible, None
        if not."""
        mean_reward_per_arms = [np.mean(self.reward_per_arms[k]) for k in range(self.K)]
        return np.argmax(mean_reward_per_arms)

    def _update_local(self, last_k, last_r):
        """Update local variables."""
        self.reward_per_arms[last_k].append(last_r)

    def act(self, t):
        """Select an arm."""
        self.tt_awake.append(t)

        if self.T_0 is not None and t < self.T_0:
            k = self.rng.integers(self.K)

        else:

            if self.rng.uniform() < self.eps:
                k = self.rng.integers(self.K)

            else:
                uu = [np.mean(self.reward_per_arms[k]) for k in range(self.K)]

                if self.random_break_tie:
                    k = self.rng.choice(np.argwhere(uu == np.amax(uu)).flatten())  # if there is a tie draw randomly
                else: k = np.argmax(uu)

        return k
