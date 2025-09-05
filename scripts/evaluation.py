import numpy as np
import torch


@torch.no_grad()
def evaluate(env, agent, episodes=20, seeds=[100, 200, 300, 400, 500]):
    total_rewards = []
    total_durations = []

    for episode in range(episodes):
        (state, _) = env.reset(seed=seeds[episodes % len(seeds)])
        truncated = terminal = False
        rewards = steps = 0
        # Episode loop
        while not (terminal or truncated):
            state = agent.norm(state)
            with torch.no_grad():
                action = agent.act(state, deterministic=True)
            newState, reward, terminal, truncated, _ = env.step(action.detach().numpy())
            state = newState
            rewards += reward
            steps += 1
        total_rewards.append(rewards)
        total_durations.append(steps)

    return np.average(total_rewards), np.average(total_durations)