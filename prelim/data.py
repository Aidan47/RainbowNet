import gymnasium as gym
import numpy as np

def collect(game):
    x = []
    y = []
    
    env = gym.make(game)
    while len(x) < 5000:
        obs, terminated, truncated = env.reset(), False, False
        
        while not terminated and not truncated and len(x) < 5000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            x.append(np.array(obs, dtype=float).reshape(-1, 1))
            y.append(games.index(game))
    return x, y


def process(x, y): # Process the data
    # Convert y index values to numpy array
    Y = np.array(y)
    
    # Shuffle examples
    shuffled_idx = np.random.permutation(len(x))
    X = np.array(x, dtype=object)[shuffled_idx]
    Y = Y[shuffled_idx]     # shuffle numpy array
    
    # Split the data into training and testing sets
    X_train, Y_train = X[:int(len(x)*0.8)], Y[:int(len(y)*0.8)]
    X_Test, Y_test = X[int(len(x)*0.8):], Y[int(len(y)*0.8):]
    
    return X_train, Y_train, X_Test, Y_test


if __name__ == "__main__":
    games = ["Humanoid-v4", "HumanoidStandup-v4", "Hopper-v4"]

    X = []
    Y = []
    for game in games:
        x, y = collect(game)
        X.extend(x)
        Y.extend(y)
        print(f"{game} collected")
        print(f"Total data points: {len(x)}")
        
    X_train, Y_train, X_test, Y_test = process(X, Y)
    
    np.savez('gameData', array1=X_train, array2=Y_train, array3=X_test, array4=Y_test)