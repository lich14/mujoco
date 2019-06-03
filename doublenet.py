import gym
import numpy as np
import os
import csv
import torch
import torch.nn as nn
import scipy
import scipy.signal
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import utils

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transition = np.dtype([('s', np.float64, (17,)), ('a', np.float64, (6,)), ('r', np.float64), ('s_', np.float64, (17,)),
                       ('a_logp', np.float64)])

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)


class store():
    buffer_capacity, batch_size = 64, 16

    def __init__(self):
        self.buffer = np.empty(self.buffer_capacity, dtype=transition)
        self.counter = 0

    def store(self, add):
        self.buffer[self.counter] = add
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False


class PPOnet(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self):
        super(PPOnet, self).__init__()
        h2 = int(np.sqrt(180 * 5))
        a2 = int(np.sqrt(180 * 60))
        self.v = nn.Sequential(
            nn.Linear(18, 180), nn.ReLU(), nn.Linear(180, h2), nn.ReLU(), nn.Linear(h2, 5), nn.ReLU(), nn.Linear(5, 1))
        self.alpha_head = nn.Sequential(
            nn.Linear(18, 180), nn.ReLU(), nn.Linear(180, a2), nn.ReLU(), nn.Linear(a2, 60), nn.ReLU(), nn.Linear(
                60, 6), nn.Softplus())
        self.beta_head = nn.Sequential(
            nn.Linear(18, 180), nn.ReLU(), nn.Linear(180, 100), nn.ReLU(), nn.Linear(100, 60), nn.ReLU(),
            nn.Linear(60, 6), nn.Softplus())

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        v = self.v(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class valuenet(nn.Module):

    def __init__(self):
        super(valuenet, self).__init__()
        h2 = int(np.sqrt(180 * 5))
        self.v = nn.Sequential(
            nn.Linear(18, 180), nn.ReLU(), nn.Linear(180, h2), nn.ReLU(), nn.Linear(h2, 5), nn.ReLU(), nn.Linear(5, 1))

    def forward(self, x):
        v = self.v(x)

        return v


class policynet(nn.Module):

    def __init__(self):
        super(policynet, self).__init__()
        a2 = int(np.sqrt(180 * 60))

        self.alpha_head = nn.Sequential(
            nn.Linear(18, 180), nn.ReLU(), nn.Linear(180, a2), nn.ReLU(), nn.Linear(a2, 60), nn.ReLU(), nn.Linear(
                60, 6), nn.Softplus())
        self.beta_head = nn.Sequential(
            nn.Linear(18, 180), nn.ReLU(), nn.Linear(180, 100), nn.ReLU(), nn.Linear(100, 60), nn.ReLU(),
            nn.Linear(60, 6), nn.Softplus())

    def forward(self, x):
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return alpha, beta


class agent():

    def __init__(self):
        self.env = gym.make('Walker2d-v2')
        self.envpoch = 1000
        self.valuenet = valuenet().double().to(device)
        self.policynet = policynet().double().to(device)
        self.optimizer_value = optim.Adam(self.valuenet.parameters(), lr=7e-5, eps=1e-7)
        self.optimizer_policy = optim.Adam(self.policynet.parameters(), lr=7e-5, eps=1e-7)
        self.clip_param = 0.2

        self.memory = store()
        self.gamma = 0.99
        self.lam = 0.95
        self.out_record = None
        self.trajectories = []

        self.path_t7 = 'model_doublenet.t7'
        self.path_lsa = "loss_doublenet_a.csv"
        self.path_lsv = "loss_doublenet_v.csv"
        self.path_ep = "episode_doublenet.csv"
        self.scaler = utils.Scaler(18)

        if os.path.isfile(self.path_t7):
            self.net.load_state_dict(torch.load(self.path_t7, map_location='cpu'))

    def storeloss(self, action_loss, value_loss):
        if os.path.isfile(self.path_lsa):
            with open(self.path_lsa, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [float(action_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)
        else:
            with open(self.path_lsa, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['loss'])
                data_row = [float(action_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)

        if os.path.isfile(self.path_lsv):
            with open(self.path_lsv, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [float(value_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)
        else:
            with open(self.path_lsv, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['loss'])
                data_row = [float(value_loss.detach().to('cpu').numpy())]
                csv_write.writerow(data_row)

    def storereward(self, episode):
        if os.path.isfile(self.path_ep):
            with open(self.path_ep, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [episode]
                csv_write.writerow(data_row)
        else:
            with open(self.path_ep, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['episode'])
                data_row = [episode]
                csv_write.writerow(data_row)

    def discount(self, x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]

    def add_gae(self, trajectories, gamma, lam):
        """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf

        Args:
            trajectories: as returned by run_policy(), must include 'values'
                key from add_value().
            gamma: reward discount
            lam: lambda (see paper).
                lam=0 : use TD residuals
                lam=1 : A =  Sum Discounted Rewards - V_hat(s)

        Returns:
            None (mutates trajectories dictionary to add 'advantages')
        """
        for trajectory in trajectories:
            if gamma < 0.999:  # don't scale for gamma ~= 1
                rewards = trajectory['rewards'] * (1 - gamma)
            else:
                rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            trg = self.discount(rewards, gamma)
            advantages = self.discount(tds, gamma * lam)
            trajectory['advantages'] = advantages
            trajectory['target_v'] = trg

    def gettraindata(self):
        observes_train = np.concatenate([t['observes'] for t in self.trajectories])
        actions_train = np.concatenate([t['actions'] for t in self.trajectories])
        advantages_train = np.concatenate([t['advantages'] for t in self.trajectories])
        old_logs = np.concatenate([t['old_log'] for t in self.trajectories])
        target_v = np.concatenate([t['target_v'] for t in self.trajectories])
        obs_re = np.concatenate([t['obs_record'] for t in self.trajectories])
        self.scaler.update(obs_re)

        advantages_train = (advantages_train - advantages_train.mean()) / (advantages_train.std() + 1e-6)

        s = torch.tensor(observes_train, dtype=torch.double).to(device)
        a = torch.tensor(actions_train, dtype=torch.double).to(device)
        adv = torch.tensor(advantages_train, dtype=torch.double).to(device)
        old_a_logp = torch.tensor(old_logs, dtype=torch.double).to(device).view(-1, 1)
        target_v = torch.tensor(target_v, dtype=torch.double).to(device)
        totalsize = observes_train.shape[0]

        return s, a, adv, old_a_logp, target_v, totalsize

    def run(self):
        for i_episode in range(10000 * self.envpoch):
            observation = self.env.reset()
            step = 0
            observes_list = []
            rewards = []
            actions = []
            values = []
            old_log = []
            if i_episode % 20 == 19:
                self.add_gae(self.trajectories, self.gamma, self.lam)
                s, a, adv, old_a_logp, target_v, totalsize = self.gettraindata()
                minibatch = 32

                for _ in range(10):
                    for index in BatchSampler(SubsetRandomSampler(range(totalsize)), minibatch, False):

                        alpha, beta = self.policynet(s[index])
                        dist = Beta(alpha, beta)
                        a_logp = dist.log_prob(a[index]).sum(dim=1)
                        ratio = torch.exp(a_logp - old_a_logp[index])
                        surr1 = ratio * adv[index]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                        action_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.smooth_l1_loss(self.valuenet(s[index]), target_v[index])
                        self.storeloss(action_loss, value_loss)
                        '''action_loss = torch.clamp(action_loss, 0, 10)
                        value_loss = torch.clamp(value_loss, 0, 10)'''

                        self.optimizer_policy.zero_grad()
                        action_loss.backward()
                        nn.utils.clip_grad_norm_(self.policynet.parameters(), 5)
                        self.optimizer_policy.step()

                        self.optimizer_value.zero_grad()
                        value_loss.backward()
                        nn.utils.clip_grad_norm_(self.valuenet.parameters(), 5)
                        self.optimizer_value.step()

                self.trajectories = []

            while (1):
                step = step + 1
                #self.env.render()

                observes = observation.astype(np.float32).reshape((1, -1))
                observes = np.append(observes, [[step]], axis=1)
                input = torch.tensor(observes, dtype=torch.double).to(device).reshape(-1, 18)
                observes_list.append(observes)

                alpha, beta = self.policynet(input)
                v = self.valuenet(input)
                values.append(v.item())
                dist = Beta(alpha, beta)
                action = dist.sample()
                actions.append(action)

                a_logp = dist.log_prob(action.view(-1, 6)).sum(dim=1)
                action = action.squeeze().cpu().numpy()
                a_logp = a_logp.item()
                old_log.append(a_logp)
                observation, reward, done, info = self.env.step(action * 2 - 1)
                rewards.append(reward)

                if done:
                    print("Episode finished after {} timesteps".format(step))
                    self.storereward(format(step))
                    obs = np.concatenate([t for t in observes_list])
                    obs_record = obs
                    scale, offset = self.scaler.get()
                    scale[-1] = 1.0  # don't scale time step feature
                    offset[-1] = 0.0  # don't offset time step feature
                    obs = (obs - offset) * scale
                    trajectory = {
                        'observes': obs,
                        'actions': np.concatenate([t.to('cpu') for t in actions]),
                        'rewards': np.array(rewards),
                        'old_log': np.array(old_log),
                        'obs_record': obs_record,
                        'values': np.array(values)
                    }

                    self.trajectories.append(trajectory)
                    break


if __name__ == "__main__":
    mujo = agent()
    mujo.run()
