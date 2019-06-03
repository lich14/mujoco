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
from torch.distributions import Beta, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import utils
import time
import argparse

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transition = np.dtype([('s', np.float64, (17,)), ('a', np.float64, (6,)), ('r', np.float64), ('s_', np.float64, (17,)),
                       ('a_logp', np.float64)])

parser = argparse.ArgumentParser(description='Train a PPO agent for warker')
parser.add_argument('--ppoepoch', type=int, default=4, help='number of ppo epochs (default: 4)')
parser.add_argument('--numminibatch', type=int, default=32, help='number of batches for ppo (default: 32)')
parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--maxgradnorm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gaelambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--usegae', action='store_true', default=True, help='use generalized advantage estimation')

args = parser.parse_args()

if os.path.exists("./csvfiles/") is False:
    os.makedirs("./csvfiles/")

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


class PPOnet(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, inputsize, actionsize, hidden_size=64):
        super(PPOnet, self).__init__()
        h2 = int(np.sqrt(inputsize * 10 * 5))
        a2 = int(np.sqrt(inputsize * 10 * actionsize * 10))
        self.act = nn.ReLU()

        self.v = nn.Sequential(
            nn.Linear(inputsize, inputsize * 10), self.act, nn.Linear(inputsize * 10, h2), self.act, nn.Linear(h2, 5),
            self.act, nn.Linear(5, 1))
        self.alpha_head = nn.Sequential(
            nn.Linear(inputsize, inputsize * 10), self.act, nn.Linear(inputsize * 10, a2), self.act,
            nn.Linear(a2, actionsize * 10), self.act, nn.Linear(actionsize * 10, actionsize), nn.Softplus())
        self.beta_head = nn.Sequential(
            nn.Linear(inputsize, inputsize * 10), self.act, nn.Linear(inputsize * 10, a2), self.act,
            nn.Linear(a2, actionsize * 10), self.act, nn.Linear(actionsize * 10, actionsize), nn.Softplus())

    def forward(self, x):
        v = self.v(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class agent():

    def __init__(self):
        self.env = gym.make('Walker2d-v2')
        self.envpoch = 1000
        self.inputsize = 17
        self.actionsize = 6
        self.net = PPOnet(self.inputsize, self.actionsize).double().to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, eps=args.eps)
        self.clip_param = 0.2
        self.PPOepoch = args.ppoepoch

        self.gamma = args.gamma
        self.lam = args.gaelambda
        self.out_record = None
        self.trajectories = []
        self.path_lsa = './csvfiles/lossa_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '.csv'
        self.path_lsv = './csvfiles/lossv_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '.csv'
        self.path_ep = './csvfiles/episode_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '.csv'
        self.scaler = utils.Scaler(18)

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
            rewards = trajectory['rewards']
            values = trajectory['values']
            # temporal differences
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            advantages = self.discount(tds, gamma * lam)
            trajectory['advantages'] = advantages
            trajectory['target_v'] = advantages + values

    def add_no_gae(self, trajectories, gamma):
        for trajectory in trajectories:
            rewards = trajectory['rewards']
            values = trajectory['values']
            tds = rewards - values + np.append(values[1:] * gamma, 0)
            trajectory['advantages'] = tds
            trajectory['target_v'] = tds + values

    def gettraindata(self):
        observes_train = np.concatenate([t['observes'] for t in self.trajectories])
        actions_train = np.concatenate([t['actions'] for t in self.trajectories])
        advantages_train = np.concatenate([t['advantages'] for t in self.trajectories])
        old_logs = np.concatenate([t['old_log'] for t in self.trajectories])
        target_v = np.concatenate([t['target_v'] for t in self.trajectories])
        '''obs_re = np.concatenate([t['obs_record'] for t in self.trajectories])
        self.scaler.update(obs_re)'''

        advantages_train = (advantages_train - advantages_train.mean()) / (advantages_train.std() + 1e-6)

        s = torch.tensor(observes_train, dtype=torch.double).to(device)
        a = torch.tensor(actions_train, dtype=torch.double).to(device)
        adv = torch.tensor(advantages_train, dtype=torch.double).to(device)
        old_a_logp = torch.tensor(old_logs, dtype=torch.double).to(device).view(-1, 1)
        target_v = torch.tensor(target_v, dtype=torch.double).to(device)

        totalsize = observes_train.shape[0]

        return s, a, adv, old_a_logp, target_v, totalsize

    def run(self):
        updatestep = 0
        update = 0
        i_episode = 0

        while (update < 100000):
            self.lr = args.lr - (args.lr * (i_episode / float(10000)))
            i_episode = i_episode + 1
            observation = self.env.reset()
            step = 0
            observes_list, rewards, actions, values, old_log = [], [], [], [], []

            if updatestep > 2048:
                update = update + 1
                updatestep = 0
                if (args.usegae):
                    self.add_gae(self.trajectories, self.gamma, self.lam)
                else:
                    self.add_no_gae(self.trajectories, self.gamma)
                s, a, adv, old_a_logp, target_v, totalsize = self.gettraindata()
                minibatch = max(totalsize // args.numminibatch, 1)

                for _ in range(self.PPOepoch):
                    for index in BatchSampler(SubsetRandomSampler(range(totalsize)), minibatch, False):

                        alpha, beta = self.net(s[index])[0]
                        dist = Beta(alpha, beta)
                        a_logp = dist.log_prob(a[index]).sum(dim=1)
                        ratio = torch.exp(a_logp - old_a_logp[index])
                        with torch.no_grad():
                            entrop = dist.entropy()

                        surr1 = ratio * adv[index]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                        action_loss = -torch.min(surr1, surr2).mean()
                        value_loss = F.mse_loss(self.net(s[index])[1], target_v[index])
                        self.storeloss(action_loss, value_loss)
                        loss = action_loss + 0.5 * value_loss - 0.01 * entrop.mean()

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.net.parameters(), args.maxgradnorm)
                        self.optimizer.step()

                self.trajectories = []

            while (1):
                step = step + 1
                updatestep = updatestep + 1
                #self.env.render()

                observes = observation.astype(np.float32).reshape((1, -1))
                input = torch.tensor(observes, dtype=torch.double).to(device).reshape(-1, self.inputsize)
                (alpha, beta), v = self.net(input)
                dist = Beta(alpha, beta)
                action = dist.sample()
                a_logp = dist.log_prob(action.view(-1, 6)).sum(dim=1)
                a_logp = a_logp.item()

                old_log.append(a_logp)
                values.append(v.item())
                observes_list.append(observes)
                actions.append(action)

                action = action.squeeze().cpu().numpy()
                observation, reward, done, info = self.env.step(action * 2 - 1)
                rewards.append(reward)

                if done:
                    print("Episode finished after {} timesteps, rewards is {}".format(step, sum(rewards)))
                    self.storereward(format(step))

                    trajectory = {
                        'observes': np.concatenate([t for t in observes_list]),
                        'actions': np.concatenate([t.to('cpu') for t in actions]),
                        'rewards': np.array(rewards),
                        'values': np.array(values),
                        'old_log': np.array(old_log)
                    }

                    self.trajectories.append(trajectory)
                    break


if __name__ == "__main__":
    mujo = agent()
    mujo.run()
