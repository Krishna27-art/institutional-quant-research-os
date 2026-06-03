"""
Reinforcement Learning for Execution

Based on Comprehensive Upgrade Analysis - Tier 2 Upgrade (#18)
Expected Sharpe improvement: +0.1–0.2

Methodology:
- PPO (Proximal Policy Optimization) for optimal execution
- Minimizes slippage and maximizes fill quality
- Trained in simulated environment with market impact
- Used by Two Sigma for execution
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch")


class ExecutionAction(Enum):
    """Execution actions"""
    WAIT = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    BUY_LARGE = 3
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6


@dataclass
class RLExecutionConfig:
    """Configuration for RL Execution Agent"""
    # Environment
    order_size: int = 10000  # Total order size
    time_steps: int = 100  # Number of execution steps
    market_impact_factor: float = 0.1  # Market impact coefficient
    
    # PPO parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Training
    n_episodes: int = 1000
    max_steps_per_episode: int = 100
    batch_size: int = 64
    ppo_epochs: int = 10
    
    # Network
    hidden_dim: int = 64
    n_layers: int = 2


class ExecutionEnvironment:
    """
    Simulated Execution Environment
    
    Simulates market conditions for training RL agent.
    """
    
    def __init__(self, config: RLExecutionConfig):
        self.config = config
        
        # State
        self.current_step = 0
        self.remaining_quantity = config.order_size
        self.executed_quantity = 0
        self.avg_fill_price = 0.0
        self.total_cost = 0.0
        
        # Market state
        self.market_price = 100.0
        self.market_volatility = 0.01
        self.order_book_depth = 1000
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode"""
        self.current_step = 0
        self.remaining_quantity = self.config.order_size
        self.executed_quantity = 0
        self.avg_fill_price = 0.0
        self.total_cost = 0.0
        
        # Randomize market conditions
        self.market_price = np.random.uniform(90, 110)
        self.market_volatility = np.random.uniform(0.005, 0.02)
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state"""
        state = np.array([
            self.remaining_quantity / self.config.order_size,
            self.executed_quantity / self.config.order_size,
            self.market_price / 100.0,
            self.market_volatility,
            self.current_step / self.config.time_steps
        ])
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return next state, reward, done, info
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Execute trade based on action
        trade_size = 0
        if action == ExecutionAction.BUY_SMALL.value:
            trade_size = min(self.remaining_quantity, 100)
        elif action == ExecutionAction.BUY_MEDIUM.value:
            trade_size = min(self.remaining_quantity, 500)
        elif action == ExecutionAction.BUY_LARGE.value:
            trade_size = min(self.remaining_quantity, 1000)
        elif action == ExecutionAction.SELL_SMALL.value:
            trade_size = -min(self.executed_quantity, 100)
        elif action == ExecutionAction.SELL_MEDIUM.value:
            trade_size = -min(self.executed_quantity, 500)
        elif action == ExecutionAction.SELL_LARGE.value:
            trade_size = -min(self.executed_quantity, 1000)
        
        # Calculate market impact
        impact = self.config.market_impact_factor * abs(trade_size) / self.order_book_depth
        fill_price = self.market_price * (1 + impact if trade_size > 0 else 1 - impact)
        
        # Update state
        if trade_size > 0:
            self.executed_quantity += trade_size
            self.remaining_quantity -= trade_size
            self.avg_fill_price = (self.avg_fill_price * (self.executed_quantity - trade_size) + 
                                  fill_price * trade_size) / self.executed_quantity
            self.total_cost += fill_price * trade_size
        elif trade_size < 0:
            self.executed_quantity += trade_size  # trade_size is negative
            self.remaining_quantity -= trade_size
            self.avg_fill_price = (self.avg_fill_price * (self.executed_quantity - trade_size) + 
                                  fill_price * abs(trade_size)) / self.executed_quantity
            self.total_cost += fill_price * abs(trade_size)
        
        # Market evolution
        self.market_price *= (1 + np.random.normal(0, self.market_volatility))
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(trade_size, impact)
        
        # Check if done
        done = (self.current_step >= self.config.time_steps or 
                self.remaining_quantity <= 0)
        
        # Info
        info = {
            "executed_quantity": self.executed_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": self.avg_fill_price,
            "total_cost": self.total_cost
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, trade_size: int, impact: float) -> float:
        """Calculate reward for action"""
        # Reward based on execution progress and cost
        progress_reward = (trade_size / self.config.order_size) * 10
        cost_penalty = -impact * 100
        
        # Penalty for not executing
        if trade_size == 0:
            progress_reward = -0.1
        
        return progress_reward + cost_penalty


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, n_layers: int):
        super(ActorCritic, self).__init__()
        
        # Shared layers
        layers = []
        input_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        
        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        shared_features = self.shared(state)
        action_logits = self.actor(shared_features)
        value = self.critic(shared_features)
        return action_logits, value
    
    def get_action(self, state):
        """Get action and log probability"""
        action_logits, value = self.forward(state)
        action_probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


class PPOExecutionAgent:
    """
    PPO Agent for Optimal Execution
    
    Trains a PPO agent to minimize slippage and maximize fill quality.
    
    Expected Sharpe improvement: +0.1–0.2
    """
    
    def __init__(self, config: RLExecutionConfig):
        self.config = config
        
        # Environment
        self.env = ExecutionEnvironment(config)
        
        # State and action dimensions
        self.state_dim = 5  # From _get_state
        self.action_dim = len(ExecutionAction)
        
        # Network
        if TORCH_AVAILABLE:
            self.actor_critic = ActorCritic(
                self.state_dim, self.action_dim, 
                config.hidden_dim, config.n_layers
            )
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.learning_rate)
        
        # Training history
        self.episode_rewards: List[float] = []
        self.episode_costs: List[float] = []
    
    def collect_trajectories(self, n_episodes: int) -> List[Dict]:
        """Collect trajectories for training"""
        trajectories = []
        
        for _ in range(n_episodes):
            state = self.env.reset()
            episode_data = []
            
            for _ in range(self.config.max_steps_per_episode):
                if TORCH_AVAILABLE:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    action, log_prob, value = self.actor_critic.get_action(state_tensor)
                    action = action.item()
                    log_prob = log_prob.item()
                    value = value.item()
                else:
                    action = np.random.randint(0, self.action_dim)
                    log_prob = 0.0
                    value = 0.0
                
                next_state, reward, done, info = self.env.step(action)
                
                episode_data.append({
                    "state": state,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": reward,
                    "done": done
                })
                
                state = next_state
                
                if done:
                    break
            
            trajectories.append(episode_data)
        
        return trajectories
    
    def compute_gae(self, rewards: List[float], values: List[float], dones: List[bool]) -> List[float]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                gae = 0
            delta = rewards[t] + self.config.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def train(self) -> Dict:
        """Train PPO agent"""
        if not TORCH_AVAILABLE:
            print("PyTorch not available, skipping training")
            return {}
        
        total_rewards = []
        
        for episode in range(self.config.n_episodes):
            # Collect trajectory
            state = self.env.reset()
            episode_data = []
            
            for step in range(self.config.max_steps_per_episode):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, log_prob, value = self.actor_critic.get_action(state_tensor)
                
                next_state, reward, done, info = self.env.step(action.item())
                
                episode_data.append({
                    "state": state,
                    "action": action.item(),
                    "log_prob": log_prob,
                    "value": value.item(),
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                })
                
                state = next_state
                
                if done:
                    break
            
            # Calculate returns and advantages
            rewards = [d["reward"] for d in episode_data]
            values = [d["value"] for d in episode_data]
            dones = [d["done"] for d in episode_data]
            
            returns = []
            R = 0
            for r, d in zip(reversed(rewards), reversed(dones)):
                if d:
                    R = 0
                R = r + self.config.gamma * R
                returns.insert(0, R)
            
            advantages = self.compute_gae(rewards, values, dones)
            
            # PPO update
            for _ in range(self.config.ppo_epochs):
                # Shuffle and batch
                indices = np.random.permutation(len(episode_data))
                
                for start in range(0, len(episode_data), self.config.batch_size):
                    end = start + self.config.batch_size
                    batch_indices = indices[start:end]
                    
                    batch_data = [episode_data[i] for i in batch_indices]
                    batch_returns = [returns[i] for i in batch_indices]
                    batch_advantages = [advantages[i] for i in batch_indices]
                    
                    # Convert to tensors
                    states = torch.FloatTensor([d["state"] for d in batch_data])
                    old_log_probs = torch.stack([d["log_prob"] for d in batch_data])
                    old_values = torch.FloatTensor([d["value"] for d in batch_data])
                    actions = torch.LongTensor([d["action"] for d in batch_data])
                    returns_tensor = torch.FloatTensor(batch_returns)
                    advantages_tensor = torch.FloatTensor(batch_advantages)
                    
                    # Forward pass
                    action_logits, values = self.actor_critic(states)
                    action_probs = torch.softmax(action_logits, dim=-1)
                    dist = Categorical(action_probs)
                    new_log_probs = dist.log_prob(actions)
                    
                    # Calculate ratio
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    
                    # PPO loss
                    surr1 = ratio * advantages_tensor
                    surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                                       1 + self.config.clip_epsilon) * advantages_tensor
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = nn.MSELoss()(values.squeeze(), returns_tensor)
                    
                    # Entropy loss
                    entropy = dist.entropy().mean()
                    
                    # Total loss
                    loss = (actor_loss + 
                           self.config.value_loss_coef * value_loss - 
                           self.config.entropy_coef * entropy)
                    
                    # Optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                    self.optimizer.step()
            
            episode_reward = sum(rewards)
            total_rewards.append(episode_reward)
            self.episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")
        
        return {
            "total_episodes": self.config.n_episodes,
            "final_avg_reward": np.mean(total_rewards[-100:]),
            "episode_rewards": total_rewards
        }
    
    def execute_order(self, order_size: int, market_price: float) -> Dict:
        """
        Execute order using trained policy
        
        Args:
            order_size: Order size
            market_price: Current market price
            
        Returns:
            Execution results
        """
        self.config.order_size = order_size
        self.env.market_price = market_price
        
        state = self.env.reset()
        executed_quantity = 0
        avg_fill_price = 0.0
        total_cost = 0.0
        
        for _ in range(self.config.max_steps_per_episode):
            if TORCH_AVAILABLE:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                action, _, _ = self.actor_critic.get_action(state_tensor)
                action = action.item()
            else:
                action = ExecutionAction.BUY_MEDIUM.value
            
            next_state, reward, done, info = self.env.step(action)
            
            executed_quantity = info["executed_quantity"]
            avg_fill_price = info["avg_fill_price"]
            total_cost = info["total_cost"]
            
            state = next_state
            
            if done:
                break
        
        return {
            "executed_quantity": executed_quantity,
            "avg_fill_price": avg_fill_price,
            "total_cost": total_cost,
            "slippage": avg_fill_price - market_price
        }


if __name__ == "__main__":
    # Example usage
    config = RLExecutionConfig(
        order_size=10000,
        time_steps=100,
        n_episodes=100,  # Reduced for testing
        hidden_dim=32
    )
    
    agent = PPOExecutionAgent(config)
    
    # Train
    print("Training PPO agent...")
    if TORCH_AVAILABLE:
        training_results = agent.train()
        print(f"\nTraining Results:")
        print(f"  Total Episodes: {training_results['total_episodes']}")
        print(f"  Final Avg Reward: {training_results['final_avg_reward']:.2f}")
    else:
        print("Skipping training (PyTorch not available)")
    
    # Execute order
    print("\nExecuting order...")
    execution_result = agent.execute_order(order_size=10000, market_price=100.0)
    print(f"\nExecution Results:")
    for key, value in execution_result.items():
        print(f"  {key}: {value}")
