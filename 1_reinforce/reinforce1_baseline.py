"""
    Modification 1: Added baseline for the returns. 
    Baseline = G.mean()
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch 
import torch.nn as nn
from torch.distributions.normal import Normal
import wandb

import gymnasium as gym


plt.rcParams["figure.figsize"] = (10, 5)

GAMMA = 0.99
LR = 0.01
HIDDEN_LAYER1 = 128

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu' 
print(f'Using device : {device}')

## Neural network
class PGNet(nn.Module):
    def __init__(self, input, fc1, n_actions):
        super().__init__()
        self.input = input 
        self.fc1 = fc1
        # self.fc2 = fc2
        self.n_actions = n_actions
        
        self.net = nn.Sequential(
            nn.Linear(self.input, self.fc1),
            nn.ReLU(), 
            nn.Linear(self.fc1, self.n_actions)
        )
        
    def forward(self, x):
        logits = self.net(x)
        return logits  
    
# calculate return
def calc_qvals(rewards):
    """
        calculate the q_val for each step, 
        g_i = r_i + gamma*(g_(i+1))
    """
    res = []
    g = 0
    for r in reversed(rewards):
        g = r + GAMMA* g
        res.append(g)

    return list(reversed(res))

env = gym.make('CartPole-v1')
eval_env = gym.make('CartPole-v1', render_mode = 'rgb_array')
run = wandb.init(
        entity = None, 
        project = "RL_diary", 
    )


policy = PGNet(input = env.observation_space.shape[0], 
               fc1 = HIDDEN_LAYER1, 
            #    fc2 = 16, 
               n_actions = env.action_space.n).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr = LR )

state, _ = env.reset()
state_list = []
reward_list = []
action_list = []


total_rewards = []
episode_idx = 0 # number of episodes = number of optimizer.step() calls

while True:
    done = False
    state_list = []
    reward_list = []
    action_list = []
    state, _ = env.reset()

        
    while not done:
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0).to(device)
        logits = policy(state)
        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample().item()

        new_state, rew, term, trunc, info  = env.step(action)
        state_list.append(state)
        reward_list.append(rew)
        action_list.append(action)
        
        if term or trunc: 
            done = True
            
        state = new_state
        
    # logging 
    episode_reward = sum(reward_list)
    total_rewards.append(episode_reward)
    mean_reward = float(np.mean(total_rewards[-100:]))
    print(f"episode : {episode_idx} | episode reward : {total_rewards[-1]} | mean reward/100 eps : {mean_reward}")
    wandb.log({
        "episode_reward": episode_reward, 
        "Mean reward/100 eps": mean_reward
    }, step=episode_idx)  
        
    # evaluation logging
    if episode_idx%100==0 and episode_idx>0:
        frames = []
        eval_state, _ = eval_env.reset()
        eval_done = False
        eval_reward = 0 
        
        while not eval_done:
            frames.append(eval_env.render())
            eval_state_tensor = torch.tensor(eval_state, dtype=torch.float32).unsqueeze(dim=0).to(device)
            with torch.no_grad():
                eval_logits = policy(eval_state_tensor)
                eval_action = torch.distributions.Categorical(logits=eval_logits).sample().item()
                
            eval_state, eval_rew, eval_term, eval_trunc, _ = eval_env.step(eval_action)
            eval_reward += eval_rew
            eval_done = eval_term or eval_trunc
            
        
        wandb.log({
            "eval_reward": eval_reward, 
            'video': wandb.Video(np.array(frames).transpose(0, 3,1,2), fps=30, format="mp4")
        }, step=episode_idx)
        
        
    G = calc_qvals(rewards=reward_list)

    # Batch all operations together
    states_batch = torch.cat(state_list, dim=0).to(device)  # Shape: (episode_length, state_dim)
    actions_batch = torch.tensor(action_list, dtype=torch.long).to(device)  # Shape: (episode_length,)
    G_tensor = torch.tensor(G, dtype=torch.float32).to(device)  # Shape: (episode_length,)
    G_tensor = G_tensor-G_tensor.mean()

    # Get logits for all states at once
    logits = policy(states_batch)
    dist = torch.distributions.Categorical(logits=logits)
    log_probs = dist.log_prob(actions_batch)

    # Compute loss in one line
    loss = -(G_tensor * log_probs).mean()  # or .sum() depending on preference
    
    wandb.log({"loss": loss.item()}, step=episode_idx)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    episode_idx += 1
    
    
                
    if mean_reward>450:
        break
   
    
wandb.finish()
env.close()
eval_env.close()
print(f"Episode finished. Steps: {len(state_list)}, Total reward: {sum(reward_list):.2f}, Loss: {loss.item():.4f}")

