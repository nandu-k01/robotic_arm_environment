
import torch.nn.functional as F
import numpy as np
import torch
import copy
import torch.nn as nn
import argparse
from datetime import datetime
import os, shutil, time
from environment import MyRLEnvironmentNode
import rclpy

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width, maxaction):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, action_dim)

        self.maxaction = maxaction

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.maxaction
        return a


class Q_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Q_Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, net_width)
        self.l2 = nn.Linear(net_width, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q

# def evaluate_policy(env, agent, turns = 3):
#     total_scores = 0
#     for j in range(turns):
#         s, info = env.reset()
#         done = False
#         while not done:
#             # Take deterministic actions at test time
#             a = agent.select_action(s, deterministic=True)
#             s_next, r, dw, tr, info = env.step(a)
#             done = (dw or tr)

#             total_scores += r
#             s = s_next
#     return int(total_scores/turns)
def evaluate_policy(env, agent, turns = 3):
    total_scores = 0
    for j in range(turns):
        s = env.reset_environment_request()
        done = False
        for k in range(1, 101):
            # Take deterministic actions at test time
            a_ev = agent.select_action(s, deterministic=True)
            # a_ev = list(a_ev)
            a_ev = list(map(float, a_ev))
            env.action_step_service(a_ev)
            r, done = env.calculate_reward_funct() 
            s_next = env.state_space_funct()
            total_scores += r
            s = s_next
    return int(total_scores/turns)


 

class DDPG_agent():
	def __init__(self, **kwargs):
		# Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
		self.__dict__.update(kwargs)
		self.tau = 0.005

		self.actor = Actor(self.state_dim, self.action_dim, self.net_width, self.max_action).to(self.dvc)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
		self.actor_target = copy.deepcopy(self.actor)

		self.q_critic = Q_Critic(self.state_dim, self.action_dim, self.net_width).to(self.dvc)
		self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.c_lr)
		self.q_critic_target = copy.deepcopy(self.q_critic)

		self.replay_buffer = ReplayBuffer(self.state_dim, self.action_dim, max_size=int(5e5), dvc=self.dvc)
		
	def select_action(self, state, deterministic):
		with torch.no_grad():
			state = torch.FloatTensor(state[np.newaxis, :]).to(self.dvc)  # from [x,x,...,x] to [[x,x,...,x]]
			a = self.actor(state).cpu().numpy()[0] # from [[x,x,...,x]] to [x,x,...,x]
			if deterministic:
				return a
			else:
				noise = np.random.normal(0, self.max_action * self.noise, size=self.action_dim)
				return (a + noise).clip(-self.max_action, self.max_action)

	def train(self):
		# Compute the target Q
		with torch.no_grad():
			s, a, r, s_next, dw = self.replay_buffer.sample(self.batch_size)
			target_a_next = self.actor_target(s_next)
			target_Q= self.q_critic_target(s_next, target_a_next)
			target_Q = r + (~dw) * self.gamma * target_Q  #dw: die or win

		# Get current Q estimates
		current_Q = self.q_critic(s, a)

		# Compute critic loss
		q_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the q_critic
		self.q_critic_optimizer.zero_grad()
		q_loss.backward()
		self.q_critic_optimizer.step()

		# Update the Actor
		a_loss = -self.q_critic(s,self.actor(s)).mean()
		self.actor_optimizer.zero_grad()
		a_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		with torch.no_grad():
			for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save(self,EnvName, timestep):
		torch.save(self.actor.state_dict(), "./model/{}_actor{}.pth".format(EnvName,timestep))
		torch.save(self.q_critic.state_dict(), "./model/{}_q_critic{}.pth".format(EnvName,timestep))

	def load(self,EnvName, timestep):
		self.actor.load_state_dict(torch.load("./model/{}_actor{}.pth".format(EnvName, timestep), map_location=self.dvc))
		self.q_critic.load_state_dict(torch.load("./model/{}_q_critic{}.pth".format(EnvName, timestep), map_location=self.dvc))


class ReplayBuffer():
    def __init__(self, state_dim, action_dim, max_size, dvc):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.a = torch.zeros((max_size, action_dim) ,dtype=torch.float,device=self.dvc)
        self.r = torch.zeros((max_size, 1) ,dtype=torch.float,device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim) ,dtype=torch.float,device=self.dvc)
        self.dw = torch.zeros((max_size, 1) ,dtype=torch.bool,device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = torch.tensor(a).to(self.dvc) # Note that a is numpy.array
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]


'''
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=400, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=1e-3, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-3, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size of training')
parser.add_argument('--random_steps', type=int, default=5e4, help='random steps before trianing')
parser.add_argument('--noise', type=float, default=0.1, help='exploring noise')
opt.state_dim = env.observation_space.shape[0]
opt.action_dim = env.action_space.shape[0]
opt.max_action = float(env.action_space.high[0])
'''   

parser = argparse.ArgumentParser()
opt = parser.parse_args()
opt.gamma = 0.99
opt.net_width = 128
opt.a_lr = 1e-3
opt.c_lr = 1e-3
opt.batch_size = 128
opt.random_steps = 5e4
opt.noise = 0.1
opt.state_dim = 12
opt.action_dim = 6
opt.max_action = 2.617
opt.seed = 0
opt.Max_train_steps = 5e6
opt.ModelIdex = 150
opt.save_interval = 1e4
opt.write = True
opt.dvc = 'cpu'
opt.Loadmodel = True
opt.eval_interval = 2e3
# opt.min_action = -2.617


def main():
    # Build Env
    EnvName = 'manipulator'
    rclpy.init(args=None)
    env = MyRLEnvironmentNode()
    rclpy.spin_once(env)
    # env = 
    # eval_env = 
    print(f'Env:{EnvName}  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{-opt.max_action} ')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    # Build SummaryWriter to record training curves
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(EnvName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)


    # Build DRL model
    if not os.path.exists('model'): os.mkdir('model')
    agent = DDPG_agent(**vars(opt)) # var: transfer argparse to dictionary

    # if opt.render:
    #     while True:
    #         score = evaluate_policy(env, agent, turns=1)
    #         print('EnvName:', BrifEnvName[opt.EnvIdex], 'score:', score)
    # else:
    # agent.load(EnvName, opt.ModelIdex)
    total_steps = 0
    while total_steps < opt.Max_train_steps:
        s = env.reset_environment_request()  # Do not use opt.seed directly, or it can overfit to opt.seed
        time.sleep(1.0)
        # env_seed += 1
        done = False

        '''Interact & train'''
        while not done:  
            if total_steps < opt.random_steps: a = env.generate_action_funct()
            else: a = agent.select_action(s, deterministic=False)
            # s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
            # a = list(a)
            a = list(map(float, a))
            env.action_step_service(a)
            r, done = env.calculate_reward_funct() 
            s_next = env.state_space_funct()
            # done = (dw or tr)

            agent.replay_buffer.add(s, a, r, s_next, done)
            s = s_next
            print(f"Total Steps: {total_steps}, rewards: {r}")
            total_steps += 1
            
            '''train'''
            if total_steps >= opt.random_steps:
                # print("started training")
                agent.train()

            if opt.write: writer.add_scalar('total_rewards', r, global_step=total_steps)

            '''record & log'''
            if total_steps % opt.eval_interval == 0:
                print("Evaluation")
                ep_r = evaluate_policy(env, agent, turns=3)
                if opt.write: writer.add_scalar('ep_r', ep_r, global_step=total_steps)
                print(f'EnvName:{EnvName}, Steps: {int(total_steps/1000)}k, Episode Reward:{ep_r}')

            '''save model'''
            if total_steps % opt.save_interval == 0:
                agent.save(EnvName, int(total_steps/1000))
        # env.close()
        # eval_env.close()



if __name__ == '__main__':
    main()



