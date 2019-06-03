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
import Utils.random_process as random_process
import utils
import copy
import argparse

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='Train a PPO agent for warker')
parser.add_argument('--ppoepoch', type=int, default=10, help='number of ppo epochs (default: 4)')
parser.add_argument('--numminibatch', type=int, default=16, help='number of batches for ppo (default: 32)')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 7e-4)')
parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--maxgradnorm', type=float, default=0.5, help='max norm of gradients (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--gaelambda', type=float, default=0.95, help='gae lambda parameter (default: 0.95)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
parser.add_argument("--tau", default=0.005, type=float)
parser.add_argument("--updata", default=256, type=float)
parser.add_argument("--exploration_noise", default=0.1, type=float)
args = parser.parse_args()
from tensorboardX import SummaryWriter
writer = SummaryWriter()

if os.path.exists("./csvfiles/") is False:
    os.makedirs("./csvfiles/")


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.act = nn.Tanh()
        inputsize = 17
        hidden_size = 400
        actionsize = 6

        self.actor = nn.Sequential(
            nn.Linear(inputsize, hidden_size), self.act, nn.Linear(hidden_size, hidden_size), self.act)

        self.mean = nn.Sequential(nn.Linear(hidden_size, actionsize), nn.Tanh())

    def forward(self, x):
        actor = self.actor(x)
        mean = self.mean(actor)
        return mean


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()

        self.act = nn.Tanh()
        inputsize = 17
        actionsize = 6
        hidden_size = 400

        self.critic = nn.Sequential(
            nn.Linear(inputsize + actionsize, hidden_size), self.act, nn.Linear(hidden_size, hidden_size), self.act)

        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, u):
        x = torch.cat([x, u], 1)
        critic = self.critic(x)
        v = self.v(critic)
        return v


class agent():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    def __init__(self):
        self.env = gym.make('Walker2d-v2')
        #self.env.seed(args.seed)
        self.envpoch = 2048
        self.inputsize = 17
        self.actionsize = 6
        self.actor = Actor().double().to(device)
        self.actor_target = Actor().double().to(device)
        self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=1e-4)

        self.critic = Critic().double().to(device)
        self.critic_target = Critic().double().to(device)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=1e-3)
        #self.random_noise = random_process.OrnsteinUhlenbeckActionNoise(self.actionsize)

        self.clip_param = 0.2
        self.PPOepoch = args.ppoepoch

        self.gamma = args.gamma
        self.lam = args.gaelambda
        self.out_record = None
        self.trainstep = 0
        self.trajectories = []

        self.path_lsa = './csvfiles/lossa_lr.csv'
        self.path_lsv = './csvfiles/lossv_lr.csv'
        self.path_ep = './csvfiles/episode_lr.csv'
        self.scaler = utils.Scaler(17)

    def getaction(self, state, noise=True):
        action = self.actor(state)
        if noise:
            action = action.data.cpu().numpy()[0] + np.random.normal(
                0, args.exploration_noise, size=6).clip(-1 * args.exploration_noise, args.exploration_noise)
        else:
            action = action.data.cpu().numpy()[0]
        action = np.clip(action, -1., 1.)
        return action

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
        rewards_train = np.concatenate([t['rewards'] for t in self.trajectories]).reshape(-1, 1)
        actions_train = np.concatenate([t['actions'] for t in self.trajectories])
        nextobserves_train = np.concatenate([t['nextobserves'] for t in self.trajectories])
        done = np.concatenate([t['done'] for t in self.trajectories]).reshape(-1, 1)

        s = torch.tensor(observes_train, dtype=torch.double).to(device)
        a = torch.tensor(actions_train, dtype=torch.double).to(device)
        r = torch.tensor(rewards_train, dtype=torch.double).to(device)
        s_ = torch.tensor(nextobserves_train, dtype=torch.double).to(device)
        done = torch.tensor(done, dtype=torch.double).to(device)
        totalsize = observes_train.shape[0]

        return s, a, totalsize, r, s_, done

    def run(self):
        updatestep = 0
        update = 0
        i_episode = 0

        while (update < 10000):
            i_episode = i_episode + 1
            observation = self.env.reset()
            step = 0
            observes_list, rewards, actions, dones = [], [], [], []

            if updatestep > args.updata:
                update = update + 1
                updatestep = 0
                #self.add_gae(self.trajectories, self.gamma, self.lam)
                s, a, totalsize, r, s_, done = self.gettraindata()
                minibatch = max(int(totalsize / args.numminibatch), 1)

                for _ in range(self.PPOepoch):
                    for index in BatchSampler(SubsetRandomSampler(range(totalsize)), minibatch, True):
                        self.trainstep += 1
                        target_Q = self.critic_target(s_[index], self.actor_target(s_[index]))
                        target_Q = r[index] + (done[index] * self.gamma * target_Q).detach()

                        current_Q = self.critic(s[index], a[index])
                        critic_loss = F.mse_loss(current_Q, target_Q)
                        self.critic_optimizer.zero_grad()
                        critic_loss.backward()
                        nn.utils.clip_grad_norm_(self.critic.parameters(), 5.)
                        self.critic_optimizer.step()

                        actor_loss = -self.critic(s[index], self.actor(s[index])).mean()
                        self.actor_optimizer.zero_grad()
                        actor_loss.backward()
                        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.)
                        self.actor_optimizer.step()
                        self.storeloss(actor_loss, critic_loss)
                        writer.add_scalar('loss/a', actor_loss, self.trainstep)
                        writer.add_scalar('loss/v', critic_loss, self.trainstep)

                        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                self.trajectories = []

            while (1):
                step = step + 1
                updatestep = updatestep + 1

                observes = observation.astype(np.float32).reshape((1, -1))
                input = torch.tensor(observes, dtype=torch.double).to(device).reshape(-1, self.inputsize)
                action = self.getaction(input)

                observes_list.append(observes)
                actions.append(action.reshape(1, -1))
                observation, reward, done, info = self.env.step(action.reshape(1, -1))
                rewards.append(reward)
                dones.append(1)

                if done:
                    print("Episode finished after {} timesteps, rewards is {}".format(step, sum(rewards)))
                    #self.storereward(format(step), sum(rewards))
                    writer.add_scalar('reward', sum(rewards), i_episode)
                    dones.append(0)
                    dones = dones[1:]
                    observes = observation.astype(np.float32).reshape((1, -1))
                    nextobs = copy.deepcopy(observes_list)
                    nextobs = nextobs[1:]
                    nextobs.append(observes)

                    trajectory = {
                        'observes': np.concatenate([t for t in observes_list]),
                        'actions': np.concatenate([t for t in actions]),
                        'rewards': np.array(rewards),
                        'nextobserves': np.concatenate([t for t in nextobs]),
                        'done': np.array(dones)
                    }

                    self.trajectories.append(trajectory)
                    break


if __name__ == "__main__":
    mujo = agent()
    mujo.run()
    writer.export_scalars_to_json("./test.json")
    writer.close()
