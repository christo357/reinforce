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


HIDDEN_LAYER1  = 256
# ALPHA = 0.95
GAMMA = 0.95 # DISCOUNT FACTOR
LAMBDA = 0.95 # FOR GAE
LR = 1e-4
# N_STEPS = 20
ENV_ID = 'InvertedDoublePendulum-v5'
N_ENVS = 4
N_STEPS = 64
BATCH_SIZE = N_ENVS * N_STEPS

ENTROPY_BETA = 0.001
ENTROPY_BETA_MIN = 1e-5
entropy_smoothing_factor = 0.05
total_updates = 500000 // BATCH_SIZE
TARGET_REWARD = 9000


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


def compute_gae(rewards, values, next_values, dones, gamma, lam):
    
    
    # print(f"REWARDS:{rewards}")
    # print(f'DONES: {dones}')
    
    mask = 1.0 - dones
    # print('MASK', mask)

    delta_t = rewards + (gamma * mask * next_values) - values
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

class VectorCollector:
    def __init__(self, envs, policy, gamma, lam, n_steps,action_low, action_high,  device):
        # super().__init__(self,)
        self.env = envs
        self.policy = policy
        self.gamma = gamma
        self.lam = lam
        self.n_steps = n_steps
        self.device = device
        
        self.ep_reward = 0
        
        self.state, _ = envs.reset()
        print(self.state)
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
            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_dones = []
            batch_values = []
            batch_deltas = []
            
            episode_rewards = []
            batch_value_next = []
            
            for _ in range(self.n_steps):
                # print(f"state: {self.state}")
                
                state_t = torch.tensor(self.state, dtype=torch.float32, device=device)#.unsqueeze(0)
                # print(state_t)
                with torch.no_grad():
                    mu, std, val = self.policy(state_t)
                # print(f"mu: {mu}")
                # print(f'std: {std}')
                # print(f"val: {val}")
                # return
                # # print('mu', mu)
                # # print('std', std)
                dist = torch.distributions.Normal(mu,std)
                u = dist.sample()
                a = torch.tanh(u)
                
                action = a*self.action_scale + self.action_bias
                # print(f"action:{action}")
                action_env = action.detach().cpu().numpy()
                # action_env = action.squeeze(0).detach().cpu().numpy()
                # print(f"action env:{action_env}")
                # return
                next_state, rew, term, trunc, info = self.env.step(action_env)
                # self.next_state_t = torch.Tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                done = term | trunc
                done_t = torch.tensor(done, dtype=torch.float32, device=self.device)
                rew_t = torch.tensor(rew, dtype=torch.float32, device=self.device)
                # print(f'next_state: {next_state}')
                # print(f"done: {done}")
                # print(f"rew: {rew}")
                # print(f"info: {info}")
                
                
                
                batch_states.append(state_t)
                batch_actions.append(u)
                batch_rewards.append(rew_t) # rew is converted to tensor to seperate each n_steps
                batch_dones.append(done_t)
                batch_values.append(val.squeeze(dim=-1))
                # print(f"batch_actions: {batch_actions}")
                # print(f"batch_states: {batch_states}")
                # print(f"batch_rewards: {batch_rewards}")
                # print(f"batch_dones: {batch_dones}")
                # print(f"batch_values : {batch_values}")
                # yield None
                # continue
                if '_episode' in info:
                    for idx, has_ep in enumerate(info['_episode']):
                        if has_ep:
                            if 'episode' in info:
                                # print(f'idx: {idx}')
                                # print(f"episode r: {info['episode']['r']}")
                                episode_rewards.append(info['episode']['r'][idx])
                # else: 
                #     episode_rewards = []
                # print(f'{episode_rewards}')
                
                self.state = next_state

                # yield None
                # continue
                
                
                
            # bootstrapping
            with torch.no_grad():
                next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device)
                _, _, nxt_val = self.policy(next_state_t)
                # print(f'next_val before squeeze dim=-1:{nxt_val}')
                nxt_val = nxt_val.squeeze(dim=-1)
                # print(f'next_val after squeeze dim=-1:{nxt_val}')
                
            T_rewards = torch.stack(batch_rewards, dim=0)
            T_dones = torch.stack(batch_dones, dim=0)
            T_values = torch.stack(batch_values, dim=0)
            
            # print(f"values_t : {T_values}, {T_values.shape}")
            # print(f'next values after unsqueeze: {nxt_val}, {nxt_val.unsqueeze(dim=0).shape}')
            T_values_next = torch.cat((T_values[1:], nxt_val.unsqueeze(dim=0)), dim=0)
            # print(f"values_t : {T_values_next}, {T_values_next.shape}")
        
            batch_adv = compute_gae(rewards=T_rewards, 
                                    values=T_values,  
                                    next_values=T_values_next, 
                                    dones=T_dones, 
                                    gamma=self.gamma, 
                                    lam=self.lam )
            
            batch_returns = batch_adv + T_values
            # print(f'batch adv: {batch_adv}')
            # print(f'batch_returns: {batch_returns}')
            
            yield {
                    'states':batch_states, 
                    'actions':batch_actions, 
                    'done':batch_dones, 
                    'adv':batch_adv,
                    'ep_rewards': episode_rewards,
                    'values':batch_values, 
                    'returns':batch_returns
            }            

def main():      
  
    run = wandb.init(
        entity=None,
        project='RL_diary', 
        config={
            'env':ENV_ID,
            "algorithm": "a2c_parallel",
            "hidden_layer": HIDDEN_LAYER1,
            'n_envs':N_ENVS, 
            'n_steps':N_STEPS,
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
    
    # env = gym.make(ENV_ID)
    envs = gym.make_vec(ENV_ID, num_envs=N_ENVS, vectorization_mode='async' )
    envs = RecordEpisodeStatistics(envs) #handles reward logging
    eval_env = gym.make(ENV_ID, render_mode='rgb_array')

            

    policy = PolicyNet(
        input_size=envs.single_observation_space.shape[0], 
        fc = HIDDEN_LAYER1, 
        action_dim=envs.single_action_space.shape[0], 
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
    # beta_scheduler = LinearBetaScheduler(
    #     beta_start=ENTROPY_BETA, 
    #     beta_end=ENTROPY_BETA_MIN, 
    #     total_steps=total_updates   # Decay fully in the first 33% of training
    # )
    beta_scheduler = BetaScheduler(
        target_reward=TARGET_REWARD, 
        beta_start=ENTROPY_BETA, 
        beta_min=ENTROPY_BETA_MIN, 
        smoothing_factor=entropy_smoothing_factor
    )

    action_low = torch.tensor(envs.single_action_space.low, dtype=torch.float32, device=device)
    action_high = torch.tensor(envs.single_action_space.high, dtype=torch.float32, device=device)
    exp_collector = VectorCollector(envs, policy, GAMMA, LAMBDA, N_STEPS,action_low, action_high, device)
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
    global_step = 0
    episode_count = 0


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

        global_step += (N_ENVS * N_STEPS)
        if len(exp['ep_rewards']) > 0:
            for ep_rew in exp['ep_rewards']:
                episode_count += 1
                
                # Update trackers
                current_beta = beta_scheduler.update(ep_rew)
                total_rewards.append(ep_rew)
                mean_reward = float(np.mean(total_rewards[-100:]))
                
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
        
        batch_states_t = torch.stack(exp['states'], dim=0)
        batch_actions_t = torch.stack(exp['actions'], dim=0)
        
        obs_dim = batch_states_t.shape[-1]
        act_dim = batch_actions_t.shape[-1]
        
        flat_states = batch_states_t.view(-1, obs_dim)
        flat_actions = batch_actions_t.view(-1, act_dim)
        flat_returns = exp['returns'].view(-1)
        flat_adv = exp['adv'].view(-1)

        mu_new, std, value = policy(flat_states)
        # print(value)
        value_t = value.squeeze(dim=-1)
        # print('values',value_t)
        # break
        
        # loss_value = F.mse_loss(value_t, returns.detach())
        #huberloss
        delta = 1.0
        loss_value = F.smooth_l1_loss(value_t,flat_returns.detach(), beta=delta)
        
        
        dist_t = torch.distributions.Normal(mu_new, std)
        logp_u = dist_t.log_prob(flat_actions).sum(dim=-1)
        a_t = torch.tanh(flat_actions)
        logp_correction = torch.log(( 1 - a_t.pow(2))+1e-6).sum(dim=-1)
        logp = logp_u - logp_correction
        
        
        flat_adv = (flat_adv - flat_adv.mean())/(flat_adv.std() + 1e-8) # normalize adv_t after returns

        loss_policy = -(logp * flat_adv.detach()).mean()
        
        
        
        entropy = dist_t.entropy().sum(dim=-1).mean()
        
        loss_total = loss_value + loss_policy - current_beta*entropy
        
        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step(mean_reward)
        
        
        
        with torch.no_grad():
            
            mu_t, std_t, v_t = policy(flat_states)
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
                        float(np.mean(flat_adv.abs().mean().item()))
                    )
        l_entropy = smooth(l_entropy, entropy.item())
        l_policy = smooth(l_policy, loss_policy.item())
        l_value = smooth(l_value, loss_value.item())
        l_total = smooth(l_total, loss_total.item())
        
        
        
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
                "kl_div": kl_div.item(),
                "mu_delta": (mu_new - mu_old).abs().mean().item(), # How much did mu shift?
                "mu_mean": mu_new.mean().item(),
                "std_mean": std.mean().item(), # Is the policy collapsing to deterministic?
                
                # --- Action Diagnostics ---
                "action_saturation": (a_t.abs() > 0.99).float().mean().item(), # CRITICAL METRIC
                "action_mean": flat_actions.mean().item(),
                "action_std": flat_actions.std().item(),
                
                # --- Gradient Diagnostics ---
                "grad_max": grad_max,
                "grad_l2": grad_means / grad_count if grad_count else 0.0,
                
                # --- Value Diagnostics ---
                "advantage_mean_abs": adv_smoothed,
                "returns_mean": flat_returns.mean().item(),
                
                
            })
        
        # wandb.log({
        #     # 'baseline':baseline,'
        #     'entropy_beta':current_beta,
        #     'advantage':adv_smoothed,
        #     'entropy':entropy,
        #     'loss_policy':l_policy,
        #     'loss_value':l_value,
        #     'loss_entropy': l_entropy, 
        #     'loss_total': l_total,
        #     'kl div': kl_div.item(),
        #     "mu_delta": (mu_new - mu_old).abs().mean().item(),
        #     "std": std.mean().item(),
        #     "adv_abs": flat_adv.abs().mean().item(),
        #     'grad_l2':grad_means/grad_count if grad_count else 0.0,
        #     'grad_max':grad_max,
        #     'batch_returns': flat_returns,
        #     "current_episode": episode_idx, 
        #     'saturation_fractions':(a_t.abs() > 0.99).float().mean().item(),
        #     'action_mean': flat_actions.mean().item(),
        #     'action_std': flat_actions.std().item(),
        #     'action_clamp_rate': (
        #         ((flat_actions <= action_low + 0.01).any(dim=-1) | 
        #         (flat_actions >= action_high - 0.01).any(dim=-1))
        #         .float().mean().item()
        #     ),
        #     'mu_mean': mu_new.mean().item(),
        #     'mu_std': mu_new.std().item(),
        #     'policy_std_mean': std.mean().item(),
        # }, step = global_step)

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
    envs.close()
# eval_env.close()

if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    main()
