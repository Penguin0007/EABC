import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

# Define base critic for ensamble
class BaseCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(BaseCritic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, sa):
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
# Define Critic as an ensamble of a list of K base critics 
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, K):
        super(Critic, self).__init__()
        self.critics = nn.ModuleList([BaseCritic(state_dim, action_dim) for _ in range(K)])

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return torch.cat([critic(sa) for critic in self.critics], dim=1)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.critics[0](sa)
    

class EABC(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        num_critics=10,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim, num_critics).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.num_critics = num_critics

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha

        self.total_it = 0


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def train(self, replay_buffer, batch_size=256, q=None):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value and target performance indicator
            target_Qs = self.critic_target(next_state, next_action)
            target_qbar = target_Qs.mean(1, keepdim=True)
            target_qstd = target_Qs.std(1, keepdim=True)
            
            target_Q = target_qbar - target_qstd
            target_Q = reward + not_done * self.discount * target_Q
            
        # Get current Q estimates and current performance indicator (to be used later in actor update)
        current_Qs = self.critic(state, action)
        current_qbar = current_Qs.mean(1, keepdim=True)
        current_qstd = current_Qs.std(1, keepdim=True)
        
        current_qt = current_Qs.quantile(q, 1, keepdim=True)
        current_upper = current_qbar + current_qstd/np.sqrt(self.num_critics)
        current_perform_indicator = (current_qt > current_upper)*1

        # Compute critic loss
        critic_loss = torch.sum((current_Qs-target_Q)**2, 1).mean()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor loss
            pi = self.actor(state)
            Qs = self.critic(state, pi)
            qbar = Qs.mean(1, keepdim=True)
            qstd = Qs.std(1, keepdim=True)
            
            Q = qbar - qstd
            lmbda = self.alpha/Q.abs().mean().detach()

            actor_loss = -lmbda * Q.mean() + (current_perform_indicator*torch.mean((pi-action)**2, 1, keepdim=True)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            return {"target_q":target_Qs, "current_q":current_Qs, "q":Qs}

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
