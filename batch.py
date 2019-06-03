import gym
import numpy as np
import os
import csv
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
transition = np.dtype([('s', np.float64, (17,)), ('a', np.float64, (6,)), ('r', np.float64), ('s_', np.float64, (17,)),
                       ('a_logp', np.float64)])

parser = argparse.ArgumentParser(description='Train a PPO agent for traffic light')
parser.add_argument('--bound', type=float, default=0)
args = parser.parse_args()


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
        self.v = nn.Sequential(nn.Linear(17, 1000), nn.ReLU(), nn.Linear(1000, 1))
        self.fc = nn.Sequential(nn.Linear(17, 1000), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(1000, 6), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(1000, 6), nn.Softplus())

        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v


class agent():

    def __init__(self):
        self.env = gym.make('Walker2d-v2')
        self.envpoch = 1000
        self.net = PPOnet().double().to(device)
        self.optimizer = optim.Adam(self.net.parameters())
        self.clip_param = 0.2
        self.PPOepoch = 4
        self.memory = store()
        self.gamma = 0.99
        self.out_record = None

        self.path_t7 = './csvfiles/model_batch.t7'
        self.path_lsa = "./csvfiles/loss_batch_a.csv"
        self.path_lsv = "./csvfiles/loss_batch_v.csv"
        self.path_ep = "./csvfiles/episode_batch.csv"

        if os.path.exists("./csvfiles/") is False:
            os.makedirs("./csvfiles/")

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

    def trainmodel(self):
        s = torch.tensor(self.memory.buffer['s'], dtype=torch.double).to(device)
        a = torch.tensor(self.memory.buffer['a'], dtype=torch.double).to(device)
        r = torch.tensor(self.memory.buffer['r'], dtype=torch.double).to(device).view(-1, 1)
        s_ = torch.tensor(self.memory.buffer['s_'], dtype=torch.double).to(device)
        r = (r - r.mean()) / (r.std() + 1e-5)
        old_a_logp = torch.tensor(self.memory.buffer['a_logp'], dtype=torch.double).to(device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]

        for _ in range(self.PPOepoch):
            for index in BatchSampler(
                    SubsetRandomSampler(range(self.memory.buffer_capacity)), self.memory.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1)
                ratio = torch.exp(a_logp - old_a_logp[index])
                with torch.no_grad():
                    entrop = dist.entropy()

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                self.storeloss(action_loss, value_loss)
                action_loss = torch.clamp(action_loss, 0, 10)
                value_loss = torch.clamp(value_loss, 0, 10)
                loss = action_loss + 2. * value_loss - args.bound * entrop.mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

        torch.save(self.net.state_dict(), self.path_t7)

    def run(self):
        for i_episode in range(10000 * self.envpoch):
            observation = self.env.reset()
            step = 0

            while (1):
                step = step + 1
                #self.env.render()
                input = torch.tensor(observation, dtype=torch.double).to(device).reshape(-1, 17)

                (alpha, beta), v = self.net(input)
                dist = Beta(alpha, beta)
                action = dist.sample()
                action_record = action
                a_logp = dist.log_prob(action.view(-1, 6)).sum(dim=1)

                action = action.squeeze().cpu().numpy()
                a_logp = a_logp.item()

                observation, reward, done, info = self.env.step(action * 2 - 1)

                ifupdata = None
                if self.out_record is not None:
                    ifupdata = self.memory.store((self.out_record[0], self.out_record[1], self.out_record[2], pict,
                                                  self.out_record[2]))
                self.out_record = [pict, action, a_logp]

                if ifupdata is True:
                    print('train')
                    self.trainmodel()

                if self.out_record is not None:

                    with torch.no_grad():
                        target_v = self.out_record[2] + self.gamma * v
                    (alpha, beta), v_c = self.net(self.out_record[0])

                    with torch.no_grad():
                        adv = target_v - v_c

                    dist = Beta(alpha, beta)
                    a_logp_update = dist.log_prob(self.out_record[1].view(-1, 6)).sum(dim=1)
                    ratio = torch.exp(a_logp_update - self.out_record[3])

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.smooth_l1_loss(v_c, target_v)

                    self.storeloss(action_loss, value_loss)
                    action_loss = torch.clamp(action_loss, 0, 10)
                    value_loss = torch.clamp(value_loss, 0, 10)
                    loss = action_loss + 2. * value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    torch.save(self.net.state_dict(), self.path_t7)

                self.out_record = [input.detach(), action_record.detach(), reward, a_logp]
                if done:
                    with torch.no_grad():
                        target_v = torch.tensor(self.out_record[2], dtype=torch.double).to(device)
                    (alpha, beta), v_c = self.net(self.out_record[0])

                    with torch.no_grad():
                        adv = target_v - v_c

                    dist = Beta(alpha, beta)
                    a_logp_update = dist.log_prob(self.out_record[1].view(-1, 6)).sum(dim=1)
                    ratio = torch.exp(a_logp_update - self.out_record[3])

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv
                    action_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.smooth_l1_loss(v_c, target_v)

                    self.storeloss(action_loss, value_loss)
                    action_loss = torch.clamp(action_loss, 0, 10)
                    value_loss = torch.clamp(value_loss, 0, 10)
                    loss = action_loss + 2. * value_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    torch.save(self.net.state_dict(), self.path_t7)
                    print("Episode finished after {} timesteps".format(step))
                    self.storereward(format(step))
                    break


if __name__ == "__main__":
    mujo = agent()
    mujo.run()
