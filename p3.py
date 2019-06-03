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
from storage import RolloutStorage
from collections import deque

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Train a PPO agent for warker')
parser.add_argument('--ppoepoch', type=int, default=4, help='number of ppo epochs (default: 4)')
parser.add_argument('--numminibatch', type=int, default=32, help='number of batches for ppo (default: 32)')
parser.add_argument('--lr', type=float, default=7e-4, help='learning rate (default: 7e-4)')
parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--maxgradnorm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
parser.add_argument('--seed', type=int, default=2, help='random seed (default: 1)')
parser.add_argument('--gaelambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--num-steps', type=int, default=20, help='number of forward steps in A2C (default: 5)')
parser.add_argument('--num-processes', type=int, default=1, help='how many training CPU processes to use (default: 1)')
args = parser.parse_args()

rollouts = RolloutStorage(args.num_steps, args.num_processes, 17, 6)

if os.path.exists("./csvfiles/") is False:
    os.makedirs("./csvfiles/")


class AddBias(nn.Module):

    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):

        bias = self._bias.t().view(1, -1)
        return x + bias


class PPOnet(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, inputsize, actionsize, hidden_size=64):
        super(PPOnet, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        self.act = nn.ReLU()

        self.actor = nn.Sequential(
            nn.Linear(inputsize, hidden_size), self.act, nn.Linear(hidden_size, hidden_size), self.act)

        self.critic = nn.Sequential(
            nn.Linear(inputsize, hidden_size), self.act, nn.Linear(hidden_size, hidden_size), self.act)

        self.mean = nn.Sequential(nn.Linear(hidden_size, actionsize), nn.Tanh())
        self.logstd = AddBias(torch.zeros(actionsize))
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        actor = self.actor(x)
        critic = self.critic(x)
        v = self.v(critic)
        mean = self.mean(actor)
        zeros = torch.tensor(np.zeros(mean.size()), dtype=torch.double).to(device)

        action_logstd = self.logstd(zeros)

        return (mean, action_logstd.exp()), v


class agent():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def __init__(self):
        self.envs = gym.make('Walker2d-v2')
        self.envs.seed(args.seed)
        self.envpoch = 2048
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
        self.path_lsa = './csvfiles/lossa_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '_gamma' + str(
            args.gamma) + '_gaelambda' + str(args.gaelambda) + '.csv'
        self.path_lsv = './csvfiles/lossv_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '_gamma' + str(
            args.gamma) + '_gaelambda' + str(args.gaelambda) + '.csv'
        self.path_ep = './csvfiles/episode_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '_gamma' + str(
            args.gamma) + '_gaelambda' + str(args.gaelambda) + '.csv'
        self.scaler = utils.Scaler(17)

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
        old_v = np.concatenate([t['values'] for t in self.trajectories])
        '''obs_re = np.concatenate([t['obs_record'] for t in self.trajectories])
        self.scaler.update(obs_re)'''
        '''self.scaler.update(observes_train)
        scale, offset = self.scaler.get()
        observes_train = (observes_train - offset) * scale'''

        advantages_train = (advantages_train - advantages_train.mean()) / (advantages_train.std() + 1e-6)

        s = torch.tensor(observes_train, dtype=torch.double).to(device)
        a = torch.tensor(actions_train, dtype=torch.double).to(device)
        adv = torch.tensor(advantages_train, dtype=torch.double).to(device)
        old_a_logp = torch.tensor(old_logs, dtype=torch.double).to(device).view(-1, 1)
        target_v = torch.tensor(old_v + advantages_train, dtype=torch.double).to(device)
        old_vs = torch.tensor(old_v, dtype=torch.double).to(device)

        totalsize = observes_train.shape[0]

        return s, a, adv, old_a_logp, target_v, totalsize, old_vs

    def run(self):
        obs = self.envs.reset()
        obs = torch.tensor(obs, dtype=torch.double).to(device)
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)

        episode_rewards = deque(maxlen=10)

        start = time.time()
        num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes
        step_lch = 0

        for j in range(num_updates):
            for step in range(args.num_steps):
                step_lch = step_lch + 1
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob = self.net(rollouts.obs[step])

                # Obser reward and next obs
                obs, reward, done, infos = self.envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])
                #print(done)
                if (done):
                    print(step_lch)
                    step_lch = 0

                # If done then clean the history of observations.
                masks = torch.FloatTensor([0.0 if done else 1.0])
                bad_masks = torch.FloatTensor([1.0])
                rollouts.insert(obs, action, action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)
            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            rollouts.after_update()

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                print(
                    "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                    .format(j, total_num_steps, int(total_num_steps / (end - start)), len(episode_rewards),
                            np.mean(episode_rewards), np.median(episode_rewards), np.min(episode_rewards),
                            np.max(episode_rewards), dist_entropy, value_loss, action_loss))


if __name__ == "__main__":
    mujo = agent()
    mujo.run()
