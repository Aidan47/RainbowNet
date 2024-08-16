import gymnasium as gym
import numpy as np

def collect(game, trials=320):
    data = []
    
    env = gym.make(game)
    while trials > 0:
        obs, terminated, truncated = env.reset(), False, False
        while not terminated and not truncated:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            data.append((obs, game))
        trials -= 1
    return np.array(data, dtype=object)


if __name__ == "__main__":
    games = ["Humanoid-v4", "HumanoidStandup-v4", "Hopper-v4"]

    data = []
    for game in games:
        data.extend(collect(game))
        print(f"{game} collected")
        print(f"Total data points: {len(data)}")
    np.save('gameData', data)