import copy
import os.path as osp
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import imageio
import wandb

class DQAgent:
    def __init__(
            self,
            args,
            env,
            net,
            device,
            action_map
    ):
        self.args = args
        self.env = env
        self.net = net
        self.target_net = copy.deepcopy(self.net)
        self.device = device
        self.action_map = action_map
        self.num_action = len(action_map)

        self.episode = self.args.episode
        self.epsilon = self.args.epsilon
        self.batch_size = self.args.batch_size
        self.gamma = torch.tensor(self.args.gamma).float().to(self.device)

        self.episode_buffer = []

        self.net_sync_freq = 1
        self.net_sync = 0

        self.fn = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)

        self.save_dir = osp.join('saveGif',str(args.rand_seed))
        if not osp.isdir(self.save_dir): os.mkdir(self.save_dir)

        self.ckpt_dir = osp.join('checkpoint', str(args.rand_seed))
        if not osp.isdir(self.ckpt_dir): os.mkdir(self.ckpt_dir)

    def _rand_max_index(self, Q):
        """
        If there are more than one max Q value, random select one
        """
        idx = []
        max_value,_ = torch.max(Q, dim=1)
        for i, q in enumerate(Q[-1]):
            if q == max_value:
                idx.append(i)
        action = np.random.choice(idx)
        return action

    def get_action(self, image, data):
        """
        Select action with epsilon random explore
        """
        Q = self.net(image, data)
        a = self._rand_max_index(Q)
        a = a if torch.rand(1,) < self.epsilon else torch.randint(0, self.num_action, (1,))
        return a

    def train(self, episode_buffer_training):
        image = torch.tensor(np.array([l[0][0] for l in episode_buffer_training])).float().to(self.device)
        image = image.permute(0, 3, 1, 2)  #[batchsize,3,224,224]
        data = torch.tensor(np.array([l[0][1] for l in episode_buffer_training])).float().to(self.device) #[b,2]
        reward = torch.tensor(np.array([l[2] for l in episode_buffer_training])).float().to(self.device)
        reward = reward.unsqueeze(1)
        image_next = torch.tensor(np.array([l[3][0] for l in episode_buffer_training])).float().to(self.device)
        image_next = image_next.permute(0,3,1,2)
        data_next = torch.tensor(np.array([l[3][1] for l in episode_buffer_training])).float().to(self.device)


        if self.net_sync_freq == self.net_sync:
            self.target_net.load_state_dict(self.net.state_dict())
            self.net_sync = 0
        else:
            self.net_sync += 1

        q = self.net(image, data)
        q = q.max(1).values.unsqueeze(1)

        with torch.no_grad():
            q_next = self.target_net(image_next, data_next)
        q_next = q_next.max(1).values.unsqueeze(1)

        target = reward + self.gamma*q_next
        self.optimizer.zero_grad()
        loss = self.fn(q, target)
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def work(self):
        reward_max = -100
        success = 0
        for i in range(self.episode):
            self.epsilon = min(self.epsilon + 0.00004, 1) # We can linear increase the epsilon
            if i % 10 == 0:
                print(f'episode:{i}')
            self.episode_buffer = []
            self.env.reset()

            image = self.env.front_view
            data = [self.env.angle, self.env.speed]
            s = [image, data]

            losses, ep_len, rew = 0, 0, 0
            while self.env.done != True:
                ep_len += 1
                image_train = torch.from_numpy(s[0].transpose((2, 0, 1))).float().unsqueeze(0).to(self.device) #numpy[224,224,3] ==> torch[1,3,224,224]
                data_train = torch.tensor(s[1]).float().unsqueeze(0).to(self.device) #torch[1,2]
                with torch.no_grad():
                    a = self.get_action(image_train, data_train)
                reward = self.env.step(self.action_map[a])
                image_n = self.env.front_view
                data_n = [self.env.angle, self.env.speed]
                sn = [image_n, data_n]
                self.episode_buffer.append([s, a, reward, sn])
                s = sn
                rew += reward
                if len(self.episode_buffer) % self.batch_size == 0 or self.env.done == True:
                    if len(self.episode_buffer) >= self.batch_size:
                        episode_buffer_training = self.episode_buffer[-self.batch_size:]
                    else:
                        episode_buffer_training = self.episode_buffer[:]
                    loss = self.train(episode_buffer_training)
                    losses += loss

                if self.env.done == True:
                    wandb.log({
                        'losses': losses,
                        'num_episode': ep_len,
                        'reward': rew
                    })
                    if rew >= 10:
                        success += 1
                    else:
                        success = 0
                    if success >= 5:
                        test = self.test()
                        if test:
                            if rew > reward_max:
                                torch.save(self.net.state_dict(), osp.join(self.ckpt_dir, 'model.pt'))
                                reward_max = rew
                        success = 0

    def test(self):
        self.env.reset()
        image = self.env.front_view
        data = [self.env.angle, self.env.speed]
        s = [image, data]
        ep_len, rew = 0, 0
        while self.env.done != True:
            ep_len += 1
            image_train = torch.from_numpy(s[0].transpose((2, 0, 1))).float().unsqueeze(0).to(self.device)
            data_train = torch.tensor(s[1]).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                Q = self.net(image_train, data_train)
                a = self._rand_max_index(Q)
            reward = self.env.step(self.action_map[a])
            image_n = self.env.front_view
            data_n = [self.env.angle, self.env.speed]
            sn = [image_n, data_n]
            s = sn
            rew += reward
        print('Episode len:{}， reward:{}'.format(ep_len,rew))

        height_front, weight_front, _ = self.env.render_frontview_episode[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_front = cv2.VideoWriter(osp.join(self.save_dir, 'test_front.mp4'), fourcc, 10,
                                      (height_front, weight_front))
        for image in self.env.render_frontview_episode:
            video_front.write(image)

        height_top, weight_top, _ = self.env.render_topview_episode[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_top = cv2.VideoWriter(osp.join(self.save_dir, 'test_top.mp4'), fourcc, 10,
                                    (height_top, weight_top))
        for image in self.env.render_topview_episode:
            video_top.write(image)

        cv2.destroyAllWindows()
        video_top.release()
        video_front.release()

        if rew >= 10:
            return True
        else:
            return False

# if __name__ == '__main__':
#     print(torch.rand(1,))
#     print(torch.randint(0, 50, (1,)))



