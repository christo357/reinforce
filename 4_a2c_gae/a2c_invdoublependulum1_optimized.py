"""
 calculate delta only at the end of a batch for efficiency. 
"""

import gymnasium as gym 
import numpy as np
import pandas as pd
import typing as tt
import torch  
import torch.nn as nn 
import torch.nn.functional as F
import wandb
from collections import deque
import os

from gymnasium.wrappers import NormalizeObservation, NormalizeReward

HIDDEN_LAYER1  = 256
# ALPHA = 0.95
GAMMA = 0.95 # DISCOUNT FACTOR
LAMBDA = 0.95 # FOR GAE
LR = 3e-4
# N_STEPS = 20
ENV_ID = 'InvertedDoublePendulum-v5'
N_ENV = 1
BATCH_SIZE = 64

ENTROPY_BETA = 0.01
ENTROPY_BETA_MIN = 1e-5
entropy_smoothing_factor = 0.05
total_updates = 500000


if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu' 
print(f'Using device : {device}')

run = wandb.init(
    entity=None,
    project='RL_diary', 
    config={
        'env':ENV_ID,
        "algorithm": "a2c",
        "hidden_layer": HIDDEN_LAYER1,
        "batch_size":  BATCH_SIZE,
        "gamma": GAMMA,
        "entropy_beta_min": ENTROPY_BETA_MIN,
        # "entropy_smoothing_factor":entropy_smoothing_factor,
        'lr':LR,
        "entropy_beta":ENTROPY_BETA,
        # "N_STEPS": N_STEPS
    }
    
)
wandb.define_metric("mean_reward_100", step_metric="episode_count")
wandb.define_metric("episode_*", step_metric="episode_count")

# Any metric starting with "loss_" or "grad_" uses 'global_step' as X-axis
wandb.define_metric("loss_*", step_metric="global_step")
wandb.define_metric("grad_*", step_metric="global_step")
wandb.define_metric("action_*", step_metric="global_step") # Catches saturation!
wandb.define_metric("mu_*", step_metric="global_step")
wandb.define_metric("std_*", step_metric="global_step")
wandb.define_metric("kl_*", step_metric="global_step")
wandb.define_metric("entropy", step_metric="global_step")

# Explicitly define the X-axes
wandb.define_metric("episode_count")
wandb.define_metric("global_step")

env = gym.make(ENV_ID)
eval_env = gym.make(ENV_ID, render_mode='rgb_array')



    
def smooth(old: tt.Optional[float], val: float, alpha: float = 0.95,) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val    

def record_video(env, policy, device, low, high, max_steps=500, ):
    """Record a single episode and return frames + reward"""
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    frame = env.render()
    if frame is not None:
        frames.append(np.array(frame, copy=True))
    while not done and steps < max_steps:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, std, _ = policy(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = torch.clamp(dist.sample(), low, high)

        state, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        total_reward += reward
        done = terminated or truncated
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame, copy=True))
        
    return frames, total_reward, steps


def compute_gae(rewards, values, next_val, dones, gamma, lam):
    # Ensure everything is 1D (N,) or (1,) to prevent shape mismatch
    # values = values.view(-1)
    # next_val = next_val.view(-1)
    # rewards = rewards.view(-1)
    # dones = dones.view(-1)
    # print('value', values)
    # print('next_val', next_val)
    # print('rewards', rewards)
    # print('dones', dones)
    # print('next_val', next_val.view(1,1))

    # Calculate T (steps)
    T = rewards.shape[0]
    
    # Concatenate current values with the bootstrap value
    # values shape becomes (N+1,)
    values = torch.cat((values, next_val), dim=0)
    # print(f'new values : {values}')

    adv = torch.zeros_like(rewards)
    gae = 0.0
    
    # 1. Loop Backwards
    for t in reversed(range(T)):
        # 2. Calculate Delta (TD Error)
        # delta = reward + gamma * V_next - V_curr
        # mask is (1-done), so if done, next value is 0
        non_terminal = 1.0 - dones[t]
        
        delta = rewards[t] + gamma * values[t+1] * non_terminal - values[t]
        
        # 3. Recursive GAE accumulation
        gae = delta + gamma * lam * non_terminal * gae
        adv[t] = gae

    return adv

# eg: 
# value tensor([ 0.0499, -0.0518, -0.1998,  0.0496,  0.0352, -0.0614,  0.0454, -0.0407,
#          0.0193,  0.0444,  0.0440,  0.0261,  0.0493,  0.0143, -0.0051,  0.0499,
#          0.0026,  0.0480,  0.0351, -0.0017,  0.0385,  0.0499, -0.0385, -0.0018,
#          0.0449, -0.0360,  0.0429,  0.0497,  0.0493,  0.0337,  0.0333,  0.0395,
#          0.0275, -0.1175,  0.0486,  0.0508,  0.0552,  0.0553, -0.0399,  0.0153,
#         -0.0552,  0.0492,  0.0372,  0.0528, -0.0084,  0.0374,  0.0314, -0.1297,
#          0.0490,  0.0187,  0.0237,  0.0663,  0.0598,  0.0497, -0.0523, -0.0460,
#         -0.1622,  0.0497, -0.0741,  0.0429,  0.0451,  0.0389,  0.0332,  0.0498],
#        device='mps:0')
# next_val tensor([0.0572], device='mps:0')
# rewards tensor([1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1.,
#         1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 1.,
#         1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 1.,
#         1., 1., 0., 1., 1., 1., 1., 1., 1., 1.], device='mps:0')
# dones tensor([0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
#         0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
#         0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
#         0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], device='mps:0')
# next_val tensor([[0.0572]], device='mps:0')
# new values : tensor([ 0.0499, -0.0518, -0.1998,  0.0496,  0.0352, -0.0614,  0.0454, -0.0407,
#          0.0193,  0.0444,  0.0440,  0.0261,  0.0493,  0.0143, -0.0051,  0.0499,
#          0.0026,  0.0480,  0.0351, -0.0017,  0.0385,  0.0499, -0.0385, -0.0018,
#          0.0449, -0.0360,  0.0429,  0.0497,  0.0493,  0.0337,  0.0333,  0.0395,
#          0.0275, -0.1175,  0.0486,  0.0508,  0.0552,  0.0553, -0.0399,  0.0153,
#         -0.0552,  0.0492,  0.0372,  0.0528, -0.0084,  0.0374,  0.0314, -0.1297,
#          0.0490,  0.0187,  0.0237,  0.0663,  0.0598,  0.0497, -0.0523, -0.0460,
#         -0.1622,  0.0497, -0.0741,  0.0429,  0.0451,  0.0389,  0.0332,  0.0498,
#          0.0572], device='mps:0')


class PolicyNet(nn.Module):
    def __init__(self, input_size, fc, action_dim, log_std_min, log_std_max):
        super().__init__()
        self.input_size = input_size
        self.fc = fc
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.fc), 
            nn.ReLU(), 
            nn.Linear(self.fc, self.fc), 
            nn.ReLU()
        )
        
        self.mu = nn.Linear(self.fc, self.action_dim)
        
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        self.critic_head = nn.Linear(self.fc, 1)
        
    def forward(self, x):
        x = self.net(x)
        
        mu = self.mu(x)
        std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
        std = std.expand_as(mu)
        
        val = self.critic_head(x)
        return mu, std, val
    
class LinearBetaScheduler:
    def __init__(self, beta_start, beta_end, total_steps):
        self.start = beta_start
        self.end = beta_end
        self.total_steps = total_steps

    def update(self, current_step):
        # Linearly decay beta based on step count
        frac = min(1.0, current_step / self.total_steps)
        return self.start + frac * (self.end - self.start)
    
class BetaScheduler:
    def __init__(self, target_reward, beta_start, beta_min=1e-4, smoothing_factor=0.01):
        self.target = target_reward
        self.start = beta_start
        self.min = beta_min
        self.alpha = smoothing_factor
        self.ema_reward = None  # Exponential Moving Average of Reward
        self.current_beta = beta_start

    def update(self, reward):
        # 1. Update EMA of Reward
        if self.ema_reward is None:
            self.ema_reward = reward
        else:
            self.ema_reward = (self.ema_reward * (1 - self.alpha)) + (reward * self.alpha)
        
        # 2. Calculate Progress (0.0 to 1.0) based on EMA
        # If ema_reward is negative, treat progress as 0
        progress = max(0.0, min(1.0, self.ema_reward / self.target))
        
        # 3. Decay Beta linearly with progress
        self.current_beta = self.start * (1.0 - progress)
        
        # 4. Clamp to minimum
        self.current_beta = max(self.current_beta, self.min)
        
        return self.current_beta

class NStepCollector:
    def __init__(self, env, policy, gamma, lam, batch_size,action_low, action_high,  device):
        # super().__init__(self,)
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.lam = lam
        self.batch_size = batch_size
        self.device = device
        
        self.ep_reward = 0
        
        self.state, _ = env.reset()
        self.action_bias = (action_high + action_low) / 2
        self.action_scale = (action_high - action_low) / 2
                
    def rollout(self):
        
        
        while True:
            
            finished_episode_rewards = []
        
            states_list = []
            actions_list = []
            rewards_list = []
            dones_list = []
            values_list = []
            
            # print(f"state: {self.state}")
            for i in range(self.batch_size):
                
                state_t = torch.tensor(self.state, dtype=torch.float32, device=device).unsqueeze(0)
            
                with torch.no_grad():
                    mu, std, val = self.policy(state_t)

                dist = torch.distributions.Normal(mu,std)
                u = dist.sample()
                a = torch.tanh(u)
                action = a*self.action_scale + self.action_bias
                action_env = action.squeeze(0).detach().cpu().numpy()
                
                next_state, rew, term, trunc, info = self.env.step(action_env)
                self.ep_reward += rew
                done = term or trunc
                
                
                states_list.append(state_t)
                actions_list.append(u)
                rewards_list.append(rew)
                dones_list.append(done)
                values_list.append(val.squeeze(0))
                
                
                if done:
                    finished_episode_rewards.append(self.ep_reward)
                    self.state, _ = self.env.reset()
                    self.ep_reward = 0
                else:
                    self.state = next_state
                
            next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            #bootstrapping
            if not term:
                with torch.no_grad():
                    _, _, v_t1 = self.policy(next_state_t)
                v_t1 = v_t1.squeeze(dim=-1)
            else:
                v_t1 = torch.zeros(1, device=self.device)
            
            batch_states = torch.cat(states_list, dim=0)
            batch_actions = torch.cat(actions_list,dim=0)
            batch_values = torch.tensor(values_list, dtype=torch.float32, device=self.device)
            batch_dones = torch.tensor(dones_list, dtype=torch.float32, device=self.device)
            batch_rewards = torch.tensor(rewards_list, dtype=torch.float32, device=self.device)
            batch_adv = compute_gae(rewards = batch_rewards, 
                                    values  = batch_values,
                                    next_val = v_t1, 
                                    dones =batch_dones, 
                                    gamma = self.gamma, 
                                    lam = self.lam)
            batch_returns = batch_adv + batch_values
            # if len(self.states)>=self.batch_size:
            yield {
                    'states':batch_states, 
                    'actions':batch_actions, 
                    'done':batch_dones, 
                    'adv':batch_adv,
                    'ep_rewards': finished_episode_rewards,
                    'values':batch_values, 
                    'returns':batch_returns
            }
                
                
policy = PolicyNet(
    env.observation_space.shape[0], 
    HIDDEN_LAYER1, 
    env.action_space.shape[0], 
    log_std_min=-20, 
    log_std_max=1,
).to(device)

optimizer = torch.optim.Adam(policy.parameters(), lr = LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=500,  # If reward doesn't go up for 500 steps, lower LR
)

current_beta = ENTROPY_BETA
beta_scheduler = LinearBetaScheduler(
    beta_start=ENTROPY_BETA, 
    beta_end=ENTROPY_BETA_MIN, 
    total_steps=total_updates   # Decay fully in the first 33% of training
)


action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
exp_collector = NStepCollector(env, policy, GAMMA, LAMBDA, BATCH_SIZE,action_low, action_high, device)
total_rewards = []
episode_idx = 0
mu_old = 0
adv_smoothed = None
l_entropy = None
l_policy = None
l_value = None
l_total = None
mean_reward = 0.0
solved = False
global_steps = 0


# print("Recording initial video (before training)...")
# initial_frames, initial_reward, initial_steps = record_video(eval_env, policy, device, low = action_low, high = action_high)
# wandb.log({
#     "video": wandb.Video(
#         np.array(initial_frames).transpose(0, 3, 1, 2), 
#         fps=30, 
#         format="mp4",
#         caption=f"Initial (untrained) - Reward: {initial_reward}, Steps: {initial_steps}"
#     ),
#     "initial_reward": initial_reward
# }, step=0)
# print(f"Initial reward: {initial_reward}, steps: {initial_steps}")


for step_idx, exp in enumerate(exp_collector.rollout()):
    # exp = exp_collector.rollout()
    # print(exp)
    # if exp is None:
    #     continue
    # break
    global_steps += BATCH_SIZE
    if exp['ep_rewards'] is not None:
        for ep_rew in exp['ep_rewards']:
            # Update Beta / Logger for EACH episode found
            current_beta = beta_scheduler.update(ep_rew)
            total_rewards.append(ep_rew)
            mean_reward = float(np.mean(total_rewards[-100:]))
            
            print(f"Episode: {episode_idx} |Steps: {step_idx} | Reward: {ep_rew} | Mean: {mean_reward:.1f}")
            
            wandb.log({
                "episode_reward": ep_rew, 
                "mean_reward_100": mean_reward,
                "episode_number": episode_idx
            }, )
            
            episode_idx += 1
            
            
        
            if mean_reward>9000:
                save_path = os.path.join(wandb.run.dir, "policy_best.pt")
                torch.save(policy.state_dict(), save_path)
                wandb.log({"best_policy_path": save_path}, step=step_idx)
                print(f"Solved! Mean reward > 9000 at episode {episode_idx}")
                solved = True
                break
        if solved: 
            break
    
    states_list = exp['states']
    raw_actions_list = exp['actions']
    done_list = exp['done']
    # deltas_list = exp['deltas']
    values_list = exp['values']
    
    
    batch_adv_t = exp['adv']
    
    # print('exp')
    batch_states_t = exp['states']
    batch_actions_t = exp['actions']
    batch_adv = exp['adv']
    batch_returns = exp['returns']


    # print(f"batch_actions: {batch_actions_t}")
    # print(f"batch_states: {batch_states_t}")
    # print(f"batch_dones: {exp['done']}")
    # print(f"batch_values : {exp['values']}")
    # print(f"batch_adv : {exp['adv']}")

    
    # break
    
    mu_new, std, value = policy(batch_states_t)
    value_t = value.squeeze(dim=1)
    # loss_value = F.mse_loss(value_t, returns.detach())
    #huberloss
    delta = 1.0
    loss_value = F.smooth_l1_loss(value_t,batch_returns.detach(), beta=delta)
    
    dist_t = torch.distributions.Normal(mu_new, std)
    logp_u = dist_t.log_prob(batch_actions_t).sum(dim=-1)
    a_t = torch.tanh(batch_actions_t)
    logp_correction = torch.log(( 1 - a_t.pow(2))+1e-6).sum(dim=-1)
    logp = logp_u - logp_correction
    
    
    batch_adv_t = (batch_adv_t - batch_adv_t.mean())/(batch_adv_t.std() + 1e-8) # normalize adv_t after returns

    loss_policy = -(logp * batch_adv_t.detach()).mean()
    
    
    
    entropy = dist_t.entropy().sum(dim=-1).mean()
    
    loss_total = loss_value + loss_policy - current_beta*entropy
    
    optimizer.zero_grad()
    loss_total.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
    optimizer.step()
    scheduler.step(mean_reward)
    
    
    
    with torch.no_grad():
        
        mu_t, std_t, v_t = policy(batch_states_t)
        new_dist_t = torch.distributions.Normal(mu_t, std_t)
        
        kl_div = torch.distributions.kl_divergence(dist_t, new_dist_t).mean()
        
    grad_max = 0.0
    grad_means = 0.0
    grad_count = 0
    for p in policy.parameters():

        grad_max = max(grad_max, p.grad.abs().max().item())
        grad_means += (p.grad ** 2).mean().sqrt().item()
        grad_count += 1
        
        
    adv_smoothed = smooth(
                    adv_smoothed,
                    float(np.mean(batch_adv_t.abs().mean().item()))
                )
    l_entropy = smooth(l_entropy, entropy.item())
    l_policy = smooth(l_policy, loss_policy.item())
    l_value = smooth(l_value, loss_value.item())
    l_total = smooth(l_total, loss_total.item())
    
    
    
    # break
    wandb.log({
        # 'baseline':baseline,
        "global_step": global_steps,
        'entropy_beta':current_beta,
        'advantage':adv_smoothed,
        'entropy':entropy,
        'loss_policy':l_policy,
        'loss_value':l_value,
        'loss_entropy': l_entropy, 
        'loss_total': l_total,
        'kl div': kl_div.item(),
        "mu_delta": (mu_new - mu_old).abs().mean().item(),
        "std": std.mean().item(),
        "adv_abs": batch_adv_t.abs().mean().item(),
        'grad_l2':grad_means/grad_count if grad_count else 0.0,
        'grad_max':grad_max,
        'batch_returns': batch_returns,
        "current_episode": episode_idx, 
        'saturation_fractions':(a_t.abs() > 0.99).float().mean().item(),
        'action_mean': batch_actions_t.mean().item(),
        'action_std': batch_actions_t.std().item(),
        'action_clamp_rate': (
            ((batch_actions_t <= action_low + 0.01).any(dim=-1) | 
            (batch_actions_t >= action_high - 0.01).any(dim=-1))
            .float().mean().item()
        ),
        'mu_mean': mu_new.mean().item(),
        'mu_std': mu_new.std().item(),
        'policy_std_mean': std.mean().item(),
    }, )
    
    # batch_raw_actions.clear()
    # batch_returns.clear()
    # batch_states.clear()
    mu_old = mu_new


    
# NEW: Record final video (after training)
print("\nRecording final video (after training)...")
final_frames, final_reward, final_steps = record_video(eval_env, policy, device, low=action_low, high =action_high)
wandb.log({
    "video": wandb.Video(
        np.array(final_frames).transpose(0, 3, 1, 2), 
        fps=30, 
        format="mp4",
        caption=f"Final (trained) - Reward: {final_reward}, Steps: {final_steps}, Episodes: {episode_idx}"
    ),
    "final_reward": final_reward
}, step=step_idx)
print(f"Final reward: {final_reward}, steps: {final_steps}")

print(f"\nTraining complete!")
print(f"Total episodes: {episode_idx}")
print(f"Total steps: {step_idx}")
print(f"Final mean reward: {mean_reward}")
    
wandb.finish()
env.close()
# eval_env.close()
