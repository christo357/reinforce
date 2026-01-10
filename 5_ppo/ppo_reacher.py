"""  
- torch.from_numpy instead of torch.tensor ( for state in rollout)

- preallocating data for b_<tensors>

- calculation of grad_max, only calculating .item() once per step( not every minibatch)

"""

import gymnasium as gym 
import numpy as np
import pandas as pd
import typing as tt
import torch  
import torch.nn as nn 
import torch.nn.functional as F
import wandb

from gymnasium.wrappers.vector import RecordEpisodeStatistics
import multiprocessing as mp
import pickle
import argparse
from collections import deque

import gc




HIDDEN_LAYER = 256
GAMMA = 0.99 # DISCOUNT FACTOR
LAMBDA = 0.95 # FOR GAE
ENV_ID = 'Reacher-v5'
N_ENVS = 4
T = 1024
MINI_BATCH_SIZE = 256
PPO_EPOCHS = 10
CLIP_EPS = 0.2

TARGET_REWARD = -3.75

# CRITIC_LR = 3e-4
# POLICY_LR = 3e-4
LR = 3e-4
ENTROPY_BETA = 0.0005
ENTROPY_BETA_MIN = 1e-8
entropy_smoothing_factor = 0.05
total_updates = 3000000 
# target_kl = 0.015




if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu' 
print(f'Using device : {device}')



    
def smooth(old: tt.Optional[float], val: float, alpha: float = 0.95,) -> float:
    if old is None:
        return val
    return old * alpha + (1-alpha)*val    

def sync_envs(training_env, eval_env):
    """
    Copies the running mean/variance from training_env to eval_env.
    """
    # 1. Get the Normalization Wrapper from the training env
    # 'obs_rms' is the object holding mean and var
    if hasattr(training_env, 'obs_rms'):
        training_obs_rms = training_env.obs_rms
    # If hidden behind RecordEpisodeStatistics, peel back one layer
    elif hasattr(training_env.env, 'obs_rms'):
        training_obs_rms = training_env.env.obs_rms
    else:
        raise AttributeError("Could not find 'obs_rms' in training_env. Ensure NormalizeObservation is used.")

    # --- 2. Get stats from Eval (Single) Env ---
    # Same logic for the evaluation environment
    if hasattr(eval_env, 'obs_rms'):
        eval_obs_rms = eval_env.obs_rms
    elif hasattr(eval_env.env, 'obs_rms'):
        eval_obs_rms = eval_env.env.obs_rms
    else:
        raise AttributeError("Could not find 'obs_rms' in eval_env. Ensure NormalizeObservation is used.")
    
    # 3. Copy the statistics
    eval_obs_rms.mean = training_obs_rms.mean.copy()
    eval_obs_rms.var = training_obs_rms.var.copy()
    eval_obs_rms.count = training_obs_rms.count


def record_video(env, policy, device, low, high, max_steps=500, ):
    """Record a single episode and return frames + reward"""
    frames = []
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    frame = env.render()
    action_bias = (high + low) / 2
    action_scale = (high - low) / 2
    if frame is not None:
        frames.append(np.array(frame, copy=True))
    while not done and steps < max_steps:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, std = policy(state_tensor)
        # dist = torch.distributions.Normal(mu, std)
        # action = torch.clamp(dist.sample(), low, high)
        a = torch.tanh(mu) # Squashing
        action = a * action_scale + action_bias

        state, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        total_reward += reward
        done = terminated or truncated
        steps += 1

        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame, copy=True))
        
    return frames, total_reward, steps

def save_checkpoint(path, policy, optimizer, envs):
    # Extract the running mean/std data
    obs_rms = envs.get_wrapper_attr('obs_rms')
    
    checkpoint = {
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Pickle the normalization statistics
        'obs_rms_mean': obs_rms.mean,
        'obs_rms_var': obs_rms.var,
        'obs_rms_count': obs_rms.count
    }
    torch.save(checkpoint, path)
    
def load_checkpoint(path, policy, envs):
    checkpoint = torch.load(path)
    policy.load_state_dict(checkpoint['model_state_dict'])
    
    # Load stats back into the environment
    obs_rms = envs.get_wrapper_attr('obs_rms')
    obs_rms.mean = checkpoint['obs_rms_mean']
    obs_rms.var = checkpoint['obs_rms_var']
    obs_rms.count = checkpoint['obs_rms_count']
    
    return policy

def compute_gae(rewards, values, next_values, dones,terms, gamma, lam):
    
    
    # print(f"REWARDS:{rewards}")
    # print(f'DONES: {dones}')
    
    mask = 1.0 - dones
    terms_mask = 1.0 - terms
    # print('MASK', mask)

    delta_t = rewards + (gamma * terms_mask * next_values) - values
    num_envs = delta_t.shape[-1] # Get the number of environments
    
    T = delta_t.shape[0]
    adv = torch.zeros_like(delta_t)
    gae = torch.zeros(num_envs, device=rewards.device)
    

    for t in reversed(range(T)):
        gae = delta_t[t] + gamma * lam * mask[t] * gae
        adv[t] = gae

    return adv


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
            nn.Tanh(), 
            nn.Linear(self.fc, self.fc), 
            nn.Tanh()
        )
        
        self.mu = nn.Linear(self.fc, self.action_dim)
        
        self.log_std = nn.Parameter(torch.zeros(self.action_dim))
        
        
    def forward(self, x):
        x = self.net(x)
        
        mu = self.mu(x)
        std = torch.exp(torch.clamp(self.log_std, self.log_std_min, self.log_std_max))
        std = std.expand_as(mu)
        
        return mu, std
    
class CriticNet(nn.Module):
    def __init__(self, input_size, fc, ):
        super().__init__()
        self.input_size = input_size
        self.fc = fc
        
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.fc), 
            # nn.LayerNorm(fc),
            nn.GELU(), 
            nn.Linear(self.fc, self.fc), 
            # nn.LayerNorm(fc),
            nn.GELU(),
            
        )
        
        
        self.critic_head = nn.Linear(self.fc, 1)
        
    def forward(self, x):
        x = self.net(x)
        
        val = self.critic_head(x)
        return val
    
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

class VectorCollector:
    def __init__(self, envs, policy, critic,  gamma, lam, n_steps,action_low, action_high,  device):
        # super().__init__(self,)
        self.env = envs
        self.policy = policy
        self.critic = critic
        self.gamma = gamma
        self.lam = lam
        self.n_steps = n_steps
        self.device = device
        
        self.ep_reward = 0
        
        self.state, _ = envs.reset()
        self.action_low = action_low
        self.action_high = action_high
        self.action_bias = (action_high + action_low) / 2
        self.action_scale = (action_high - action_low) / 2
                
        self.states = []
        self.raw_actions = []
        self.rewards = []
        self.dones = []
        self.next_states = []
        self.deltas = []
        self.values = []         


    def rollout(self):
        
        while True:
            
            
            episode_rewards = []
            
            
            obs_dim = self.env.single_observation_space.shape[0]
            act_dim = self.env.single_action_space.shape[0]
            n_envs = self.env.num_envs

            states_buf = torch.empty((self.n_steps, n_envs, obs_dim), device=self.device, dtype=torch.float32)
            actions_buf = torch.empty((self.n_steps, n_envs, act_dim), device=self.device, dtype=torch.float32)
            env_actions_buf = torch.empty((self.n_steps, n_envs, act_dim), device=self.device, dtype=torch.float32)
            rewards_buf = torch.empty((self.n_steps, n_envs), device=self.device, dtype=torch.float32)
            dones_buf = torch.empty((self.n_steps, n_envs), device=self.device, dtype=torch.float32)
            terms_buf = torch.empty((self.n_steps, n_envs), device=self.device, dtype=torch.float32)
            values_buf = torch.empty((self.n_steps, n_envs), device=self.device, dtype=torch.float32)
            logp_buf = torch.empty((self.n_steps, n_envs), device=self.device, dtype=torch.float32)

            for t in range(self.n_steps):
                # self.state is a numpy array of shape (N_ENVS, obs_dim)
                # Use from_numpy to avoid extra copies on CPU, then move to device.
                state_t = torch.from_numpy(self.state).to(device=self.device, dtype=torch.float32)

                with torch.no_grad():
                    mu, std = self.policy(state_t)
                    val = self.critic(state_t).squeeze(dim=-1)
                # print(f"mu: {mu}")
                # print(f'std: {std}')
                # print(f"val: {val}")
                # return
                # # print('mu', mu)
                # # print('std', std)
                dist = torch.distributions.Normal(mu,std)
                u = dist.sample()
                
                # action = torch.clamp(u, self.action_low, self.action_high )
                a = torch.tanh(u)
                action = a*self.action_scale + self.action_bias
                
                
                # print(f"action:{action}")
                # print(f"log_prob : {log_prob}")
                # print(f"log_prob : {log_prob.sum(dim=-1)}")
                # yield None
                action_env = action.detach().cpu().numpy()
                next_state, rew, term, trunc, info = self.env.step(action_env)
                # self.next_state_t = torch.Tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                done = term | trunc
                
                
                # Log probs of raw u
                log_prob_u = dist.log_prob(u).sum(dim=-1)  # (N_ENVS,)
                # # logp_correction = 2 * (np.log(2) - a - F.softplus(-2 * a)).sum(dim=-1)
                logp_correction = torch.log(( 1 - a.pow(2))+1e-6).sum(dim=-1)
                log_prob = log_prob_u - logp_correction

                # Write buffers
                states_buf[t].copy_(state_t)
                actions_buf[t].copy_(u)
                env_actions_buf[t].copy_(action)
                rewards_buf[t].copy_(torch.as_tensor(rew, device=self.device, dtype=torch.float32))
                dones_buf[t].copy_(torch.as_tensor(done, device=self.device, dtype=torch.float32))
                terms_buf[t].copy_(torch.as_tensor(term, device=self.device, dtype=torch.float32))
                values_buf[t].copy_(val)
                logp_buf[t].copy_(log_prob)

                # Episode stats from RecordEpisodeStatistics
                
                
                
                # done_t = torch.tensor(done, dtype=torch.float32, device=self.device)
                # rew_t = torch.tensor(rew, dtype=torch.float32, device=self.device)
                
                # print(f'next_state: {next_state}')
                # print(f"done: {done}")
                # print(f"rew: {rew}")
                # print(f"info: {info}")
                
                
                
                # print(f"batch_actions: {batch_actions}")
                # print(f"batch_states: {batch_states}")
                # print(f"batch_rewards: {batch_rewards}")
                # print(f"batch_dones: {batch_dones}")
                # print(f"batch_values : {batch_values}")
                # yield None
                # continue
                if '_episode' in info:
                    for idx, has_ep in enumerate(info['_episode']):
                        if has_ep and ( 'episode' in info):
                                # print(f'idx: {idx}')
                                # print(f"episode r: {info['episode']['r']}")
                                episode_rewards.append(info['episode']['r'][idx])
                
                
                self.state = next_state

                # yield None
                # continue
                
                
                
            # bootstrapping
            with torch.no_grad():
                next_state_t = torch.from_numpy(next_state).to(device=device, dtype=torch.float32)
                nxt_val = self.critic(next_state_t).squeeze(dim=-1)
                
            values_next = torch.empty_like(values_buf)
            values_next[:-1].copy_(values_buf[1:])
            values_next[-1].copy_(nxt_val)
            
            
        
            batch_adv = compute_gae(
                rewards=rewards_buf,
                values=values_buf, 
                next_values=values_next, 
                dones=dones_buf,
                terms=terms_buf,
                gamma=self.gamma,
                lam=self.lam,
                
            )
                                    
            
            batch_returns = batch_adv + values_buf
            # print(f'batch adv: {batch_adv}')
            # print(f'batch_returns: {batch_returns}')
            
            yield {
                'states': states_buf,
                'actions': actions_buf,
                'env_actions':env_actions_buf,
                'done': dones_buf,
                'adv': batch_adv,
                'ep_rewards': episode_rewards,
                'values': values_buf,
                'returns': batch_returns,
                'log_probs': logp_buf,
            }           

def main():      
    # parser = argparse.ArgumentParser()

    # # 2. Add arguments
    # # A required positional argument
    # parser.add_argument('env_id', type=str, help='Gym environment to use to run the code')

    # # 3. Parse the arguments from the command line
    # args = parser.parse_args()

    # # 4. Access and use the arguments
    # if args.verbose:
    #     print(f"Verbose mode enabled. Processing file: {args.filename}")
    # else:
    #     print(f"Processing file: {args.filename}")
    
    save_path = f"{ENV_ID}_best.pth"

  
    run = wandb.init(
        entity=None,
        project='RL_diary', 
        config={
            'env':ENV_ID,
            "algorithm": "PPO",
            "hidden_layer": HIDDEN_LAYER,
            'n_envs':N_ENVS, 
            'horizon':T,
            "mini_batch_size":  MINI_BATCH_SIZE,
            "gamma": GAMMA,
            # "entropy_beta_min": ENTROPY_BETA_MIN,
            # "entropy_smoothing_factor":entropy_smoothing_factor,
            # 'policy_lr':POLICY_LR,
            # 'critic_lr':CRITIC_LR,
            'lr':LR,
            "entropy_beta":ENTROPY_BETA,
            'clip_eps':CLIP_EPS,
            # "T": T
        }
    )
    
    # Save the code to wandb
    wandb.save(__file__, policy='now')
    
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
    
    # env = gym.make(ENV_ID)
    envs = gym.make_vec(ENV_ID, num_envs=N_ENVS, vectorization_mode='sync' )
    
    envs = RecordEpisodeStatistics(envs)
    envs = gym.wrappers.vector.NormalizeObservation(envs) 
    # envs = gym.wrappers.vector.NormalizeReward(envs)
    
    
    
    eval_env = gym.make(ENV_ID, render_mode='rgb_array')
    eval_env = gym.wrappers.NormalizeObservation(eval_env)
            

    policy = PolicyNet(
        input_size=envs.single_observation_space.shape[0], 
        fc = HIDDEN_LAYER, 
        action_dim=envs.single_action_space.shape[0], 
        log_std_min=-20, 
        log_std_max=2,
    ).to(device)
    critic = CriticNet(
        input_size=envs.single_observation_space.shape[0], 
        fc = HIDDEN_LAYER, 
    ).to(device)

    lr_lambda = lambda update: 1.0 - (update / total_updates)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr = LR)
    policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer, T_max=total_updates, eta_min=LR * 0.05
    )
    # policy_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     policy_optimizer, 
    #     mode='max', 
    #     factor=0.5, 
    #     patience=1000,  # If reward doesn't go up for 500 steps, lower LR
    # )

    critic_optimizer = torch.optim.Adam(critic.parameters(), lr = LR)
    critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer, T_max=total_updates, eta_min=LR * 0.05
    )
    # critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     critic_optimizer, 
    #     mode='max', 
    #     factor=0.5, 
    #     patience=1000,  # If reward doesn't go up for 500 steps, lower LR
    # )
    current_beta = ENTROPY_BETA
    beta_scheduler = LinearBetaScheduler(
        beta_start=ENTROPY_BETA, 
        beta_end=ENTROPY_BETA_MIN, 
        total_steps=total_updates   # Decay fully in the first 33% of training
    )
    
    # beta_scheduler = BetaScheduler(
    #     target_reward=TARGET_REWARD, 
    #     beta_start=ENTROPY_BETA, 
    #     beta_min=ENTROPY_BETA_MIN, 
    #     smoothing_factor=entropy_smoothing_factor
    # )

    action_low = torch.tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(envs.single_action_space.high, dtype=torch.float32, device=device)
    exp_collector = VectorCollector(envs, policy, critic, GAMMA, LAMBDA, T,action_low, action_high, device)
    total_rewards = deque(maxlen=100)
    episode_idx = 0
    adv_smoothed = None
    l_entropy = None
    l_policy = None
    l_value = None
    l_total = None
    mean_reward = 0.0
    solved = False
    global_step = 0
    episode_count = 0
    batch_size = N_ENVS * T


    print("Recording initial video (before training)...")
    print("Syncing normalization stats...")
    sync_envs(envs, eval_env)
    initial_frames, initial_reward, initial_steps = record_video(eval_env, policy, device, low = action_low, high = action_high)
    wandb.log({
        "video": wandb.Video(
            np.array(initial_frames).transpose(0, 3, 1, 2), 
            fps=30, 
            format="mp4",
            caption=f"Initial (untrained) - Reward: {initial_reward}, Steps: {initial_steps}"
        ),
        "initial_reward": initial_reward
    }, step=0)
    print(f"Initial reward: {initial_reward}, steps: {initial_steps}")


    for step_idx, exp in enumerate(exp_collector.rollout()):
        
        global_step += (batch_size)
        
        # break
        if len(exp['ep_rewards']) > 0:
            for ep_rew in exp['ep_rewards']:
                episode_count += 1
                
                total_rewards.append(ep_rew)
                mean_reward = float(np.mean(total_rewards))
                
                print(f"Update: {global_step} | Step: {step_idx} | Episode: {episode_count} | Mean Reward: {mean_reward:.1f}")

                # Log independent event for each finished episode
                # We DO NOT pass 'step=step_idx' here to avoid clumping data points
                wandb.log({
                    "episode_reward": ep_rew,
                    "mean_reward_100": mean_reward,
                    "episode_count":episode_count
                })

                if mean_reward > TARGET_REWARD:
                    solved = True
            
            if solved: break
        
        b_states = exp['states'].reshape(-1, envs.single_observation_space.shape[0])
        b_actions = exp['actions'].reshape(-1, envs.single_action_space.shape[0])
        b_env_actions = exp['env_actions'].reshape(-1, envs.single_action_space.shape[0])
        b_log_probs = exp['log_probs'].reshape(-1)
        b_returns = exp['returns'].reshape(-1)
        b_values = exp['values'].reshape(-1)
        b_advs = exp['adv'].reshape(-1)
        
        b_advs = (b_advs - b_advs.mean())/(b_advs.std() + 1e-8)
        
        del exp #free up memory
        
        idxs = np.arange(batch_size)
        clip_fracs = []
        approx_kls = []
        
        batch_mu = []
        batch_std = []
        
        for i in range(PPO_EPOCHS):
            np.random.shuffle(idxs)
            
            grad_max_t = torch.zeros((), device=device)   # scalar tensor
            grad_rms_sum_t = torch.zeros((), device=device)
            grad_count = 0

            
            for start in range(0, batch_size, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                mb_idxs = idxs[start:end]
        
        
                mb_states = b_states[mb_idxs]
                mb_actions = b_actions[mb_idxs]
                mb_log_probs = b_log_probs[mb_idxs]
                mb_returns = b_returns[mb_idxs]
                mb_advs = b_advs[mb_idxs]
                mb_vals = b_values[mb_idxs]
                
                
                mu_new, std_new = policy(mb_states)
                dist_new = torch.distributions.Normal(mu_new, std_new)
                log_prob_u = dist_new.log_prob(mb_actions).sum(dim=-1)
                value_new = critic(mb_states).squeeze(dim=-1)
                
                
                a_t = torch.tanh(mb_actions)
                # logp_correction = 2 * (np.log(2) - a_t - F.softplus(-2 * a_t)).sum(dim=-1)
                logp_correction = torch.log(( 1 - a_t.pow(2))+1e-6).sum(dim=-1)
                log_prob_new = log_prob_u - logp_correction
                
                logratio = log_prob_new - mb_log_probs 
                ratio = torch.exp(logratio)
                
                entropy = dist_new.entropy().sum(dim=-1).mean()
                
                # --- CALCULATE APPROX KL (For logging/early stopping) ---
                with torch.no_grad():
                    # http://joschu.net/blog/kl-approx.html
                    approx_kl = ((ratio - 1) - logratio).mean()
                    approx_kls.append(approx_kl.item())
                    clip_fracs.append(((ratio - 1.0).abs() > CLIP_EPS).float().mean().item())
                    
                    # store mu, std for logging
                    batch_mu.append(mu_new.mean().item())
                    batch_std.append(std_new.mean().item())
                
                
                # if approx_kl > target_kl:
                #     break  # stop minibatches
                
                #surrogates
                pg_loss1 = - mb_advs * ratio
                pg_loss2 = - mb_advs * torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) 
                loss_policy = torch.max(pg_loss1, pg_loss2).mean() #- (current_beta * entropy)
                loss_total = torch.max(pg_loss1, pg_loss2).mean() - (current_beta * entropy)
                
                policy_optimizer.zero_grad()
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                policy_optimizer.step()
                policy_scheduler.step()
                
                # loss_value = F.smooth_l1_loss(value_new,mb_returns.detach(), beta=1.0)
                # loss_value = F.mse_loss(value_new, mb_returns.detach())
                
                # --- VALUE CLIPPING ---
                v_pred = value_new
                v_target = mb_returns.detach()

                # 1. Unclipped Loss (Standard MSE)
                # We square the difference first
                v_loss_unclipped = (v_pred - v_target) ** 2

                # 2. Clipped Value
                # We constrain how much the new value can deviate from the OLD value
                v_clipped = mb_vals + torch.clamp(
                    v_pred - mb_vals, 
                    -CLIP_EPS, 
                    CLIP_EPS
                )
                v_loss_clipped = (v_clipped - v_target) ** 2

                # 3. Max Loss (Pessimistic bound)
                # We take the max (worst case) of the clipped vs unclipped loss, then mean it.
                # The 0.5 factor is standard for MSE (derivative of x^2 is 2x, so 0.5 cancels the 2).
                loss_value = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                                
                
                critic_optimizer.zero_grad()
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                critic_optimizer.step()
                critic_scheduler.step()
        
                
                
                with torch.no_grad():
                    for p in policy.parameters():
                        if p.grad is None:
                            continue

                        g = p.grad

                        # max |grad|
                        grad_max_t = torch.maximum(grad_max_t, g.abs().max())

                        # RMS (sqrt(mean(g^2))) per-parameter, accumulate
                        grad_rms_sum_t += g.pow(2).mean().sqrt()
                        grad_count += 1
            
                
                    
                    
                adv_smoothed = smooth(
                                adv_smoothed,
                                float(np.mean(mb_advs.abs().mean().item()))
                            )
                l_entropy = smooth(l_entropy, entropy.item())
                l_policy = smooth(l_policy, loss_policy.item())
                l_value = smooth(l_value, loss_value.item())
                l_total = smooth(l_total, loss_policy.item()+loss_value.item())
        
        # if approx_kl > target_kl:
        #     break      # stop epochs
        
        # once per UPDATE (outside minibatch loops) , take .item() only once per logging
        grad_max = grad_max_t.item()
        grad_l2 = (grad_rms_sum_t / max(1, grad_count)).item()
        
        # Calculate Explained Variance
        # It tells you if the Critic is actually learning (should be close to 1.0, not 0 or negative)
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        
        # break
        # print(f"Episode: {episode_idx} |Steps: {step_idx} | Reward: {ep_rew} | Mean: {mean_reward:.1f}")
        wandb.log({
                "global_step": global_step,  
                
                # --- Loss & entropy ---
                "loss_policy": l_policy,
                "loss_value": l_value,
                "loss_total": l_total,
                "entropy": l_entropy,
                "entropy_beta": current_beta,
                
                # --- Policy Diagnostics ---
                "kl_div": np.mean(approx_kls),
                "clip_fracs":np.mean(clip_fracs),
                "explained_variance":explained_var,
                
                # --- Policy Diagnostics (Use the FULL BATCH 'b_' variables) ---
                # Using b_states/b_actions gives you stats for the whole 2048 steps, 
                "mu_mean": np.mean(batch_mu),      
                "std_mean": np.mean(batch_std), # Is the policy collapsing to deterministic?
                
                # --- Action Diagnostics ---
                "action_saturation": (b_env_actions.abs() > 0.99).float().mean().item(), # saturation in env action space
                "action_mean": b_env_actions.mean().item(),
                "action_std": b_env_actions.std().item(),
                
                # --- Gradient Diagnostics ---
                "grad_max": grad_max,
                "grad_l2": grad_l2,
                
                # --- Value Diagnostics ---
                "advantage_mean_abs": adv_smoothed,
                "returns_mean": b_returns.mean().item(),
                
                
        })
        
        del b_states, b_actions, b_env_actions, b_log_probs, b_returns, b_values, b_advs
        del clip_fracs, approx_kls, batch_mu, batch_std
        # empty_device_cache(device)
        
        # Update trackers
        current_beta = beta_scheduler.update(global_step)

        
    # NEW: Record final video (after training)
    print("\nRecording final video (after training)...")
    print("Syncing normalization stats...")
    sync_envs(envs, eval_env)
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
    envs.close()
# eval_env.close()

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
