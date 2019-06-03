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
import argparse
from tensorboardX import SummaryWriter
writer = SummaryWriter()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Train a PPO agent for warker')
parser.add_argument('--ppoepoch', type=int, default=10, help='number of ppo epochs (default: 4)')
parser.add_argument('--numminibatch', type=int, default=32, help='number of batches for ppo (default: 32)')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 7e-4)')
parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--maxgradnorm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gaelambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
args = parser.parse_args()

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
        '''h2 = int(np.sqrt(inputsize * 10 * 5))
        a2 = int(np.sqrt(inputsize * 10 * valuesize * 10))
        self.act = nn.ReLU()

        self.v = nn.Sequential(
            nn.Linear(inputsize, inputsize * 10), self.act, nn.Linear(inputsize * 10, h2), self.act, nn.Linear(h2, 5),
            self.act, nn.Linear(5, 1))
        self.alpha_head = nn.Sequential(
            nn.Linear(inputsize, inputsize * 10), self.act, nn.Linear(inputsize * 10, a2), self.act,
            nn.Linear(a2, valuesize * 10), self.act, nn.Linear(valuesize * 10, valuesize), nn.Softplus())
        self.beta_head = nn.Sequential(
            nn.Linear(inputsize, inputsize * 10), self.act, nn.Linear(inputsize * 10, a2), self.act,
            nn.Linear(a2, valuesize * 10), self.act, nn.Linear(valuesize * 10, valuesize), nn.Softplus())'''

        self.act = nn.Tanh()

        self.actor = nn.Sequential(
            nn.Linear(inputsize, hidden_size), self.act, nn.Linear(hidden_size, hidden_size), self.act)

        self.critic = nn.Sequential(
            nn.Linear(inputsize, hidden_size), self.act, nn.Linear(hidden_size, hidden_size), self.act)
        '''self.alpha_head = nn.Sequential(nn.Linear(hidden_size, actionsize), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_size, actionsize), nn.Softplus())'''

        self.mean = nn.Sequential(nn.Linear(hidden_size, actionsize))
        #self.mean = nn.Linear(hidden_size, actionsize)
        self.logstd = AddBias(torch.zeros(actionsize))
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x):
        actor = self.actor(x)
        critic = self.critic(x)
        v = self.v(critic)
        '''alpha = self.alpha_head(critic) + 1
        beta = self.beta_head(critic) + 1'''

        mean = self.mean(actor)
        zeros = torch.tensor(np.zeros(mean.size()), dtype=torch.double).to(device)

        action_logstd = self.logstd(zeros)

        return (mean, action_logstd.exp()), v


class agent():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def __init__(self):
        self.env = gym.make('Walker2d-v2')
        self.env.seed(args.seed)
        self.envpoch = 2048
        self.inputsize = 17
        self.actionsize = 6
        self.net = PPOnet(self.inputsize, self.actionsize).double().to(device)
        #self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr, eps=args.eps)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=args.lr)
        self.optimizer = optim.Adam(self.net.parameters())
        self.clip_param = 0.2
        self.PPOepoch = args.ppoepoch
        self.trainstep = 0

        self.gamma = args.gamma
        self.lam = args.gaelambda
        self.out_record = None
        self.trajectories = []
        '''self.path_lsa = './csvfiles/lossa_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '_gamma' + str(
            args.gamma) + '_gaelambda' + str(args.gaelambda) + '.csv'
        self.path_lsv = './csvfiles/lossv_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '_gamma' + str(
            args.gamma) + '_gaelambda' + str(args.gaelambda) + '.csv'
        self.path_ep = './csvfiles/episode_lr' + str(args.lr) + '_ppoepoch' + str(args.ppoepoch) + '_gamma' + str(
            args.gamma) + '_gaelambda' + str(args.gaelambda) + '.csv'''

        self.path_lsa = './csvfiles/lossa_lr.csv'
        self.path_lsv = './csvfiles/lossv_lr.csv'
        self.path_ep = './csvfiles/episode_lr.csv'
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

    def storereward(self, episode, reward):
        if os.path.isfile(self.path_ep):
            with open(self.path_ep, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [episode, reward]
                csv_write.writerow(data_row)
        else:
            with open(self.path_ep, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['episode'])
                data_row = [episode, reward]
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
        adv = torch.tensor(advantages_train, dtype=torch.double).to(device).reshape(-1, 1)
        old_a_logp = torch.tensor(old_logs, dtype=torch.double).to(device).view(-1, 1)
        #target_v = torch.tensor(advantages_train + old_v, dtype=torch.double).to(device)
        target_v = torch.tensor(target_v, dtype=torch.double).to(device)
        old_vs = torch.tensor(old_v, dtype=torch.double).to(device)

        totalsize = observes_train.shape[0]
        target_v = target_v.reshape(-1, 1)

        return s, a, adv, old_a_logp, target_v, totalsize, old_vs

    def run(self):
        updatestep = 0
        update = 0
        i_episode = 0

        while (update < 10000):
            i_episode = i_episode + 1
            observation = self.env.reset()
            step = 0
            observes_list, rewards, actions, values, old_log = [], [], [], [], []

            if updatestep > 2048:
                self.lr = args.lr - (args.lr * (i_episode / float(10000)))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr
                update = update + 1
                updatestep = 0
                self.add_gae(self.trajectories, self.gamma, self.lam)
                #self.add_no_gae(self.trajectories, self.gamma)
                s, a, adv, old_a_logp, target_v, totalsize, old_vs = self.gettraindata()
                minibatch = max(int(totalsize / args.numminibatch), 1)

                for _ in range(self.PPOepoch):
                    for index in BatchSampler(SubsetRandomSampler(range(totalsize)), minibatch, True):
                        self.trainstep += 1
                        (mean, std), v = self.net(s[index])
                        dist = Normal(mean, std)
                        a_logp = dist.log_prob(a[index]).sum(-1, keepdim=True)

                        ratio = torch.exp(a_logp - old_a_logp[index])
                        with torch.no_grad():
                            entrop = dist.entropy()

                        surr1 = ratio * adv[index]
                        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                        action_loss = -torch.min(surr1, surr2).mean()

                        #value_loss = 0.5 * (target_v[index] - self.net(s[index])[1]).pow(2).mean()
                        '''value_pred_clipped = old_vs[index] + (v - old_vs[index]).clamp(
                            -self.clip_param, self.clip_param)
                        value_losses = (v - target_v[index]).pow(2)
                        value_losses_clipped = (value_pred_clipped - target_v[index]).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()'''

                        value_loss = F.l1_loss(target_v[index], v)
                        #self.storeloss(action_loss, value_loss)
                        #action_loss = torch.clamp(action_loss, 0, 10)
                        #value_loss = torch.clamp(value_loss, 0, 10)
                        '''if action_loss > 10 or value_loss > 10:
                            break'''
                        loss = action_loss + value_loss - 0.0 * entrop.mean()

                        self.optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.net.parameters(), args.maxgradnorm)
                        self.optimizer.step()
                        writer.add_scalar('loss/a', action_loss, self.trainstep)
                        writer.add_scalar('loss/v', value_loss, self.trainstep)

                self.trajectories = []

            while (1):
                step = step + 1
                updatestep = updatestep + 1
                #self.env.render()

                observes = observation.astype(np.float32).reshape((1, -1))
                input = torch.tensor(observes, dtype=torch.double).to(device).reshape(-1, self.inputsize)
                (mean, std), v = self.net(input)
                dist = Normal(mean, std)
                action = dist.sample()
                a_logp = dist.log_prob(action.view(-1, 6)).sum(dim=1)
                a_logp = a_logp.item()

                old_log.append(a_logp)
                values.append(v.item())
                observes_list.append(observes)
                actions.append(action)

                action = action.squeeze().cpu().numpy()
                observation, reward, done, info = self.env.step(action)
                rewards.append(reward)

                if done:
                    print("Episode finished after {} timesteps, rewards is {}".format(step, sum(rewards)))
                    writer.add_scalar('reward', sum(rewards), i_episode)
                    #self.storereward(format(step), sum(rewards))
                    observes = observation.astype(np.float32).reshape((1, -1))
                    input = torch.tensor(observes, dtype=torch.double).to(device).reshape(-1, self.inputsize)
                    (mean, std), v = self.net(input)
                    #values.append(v.item())

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
    writer.export_scalars_to_json("./test.json")
    writer.close()
