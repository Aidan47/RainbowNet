import csv, os, time, pickle, torch, numpy as np
from algo.components.actor import Actor
from algo.components.critic import Critic

class CSVLogger:
    def __init__(self, algo, env):
        self.algo = algo
        self.saved = False
        self.path = self.get_path(env, algo)
        self.fieldnames = None
        self._writer = None

    def get_path(self, env, algo, run=0):
        base = f"checkpoints/{algo}/{env}/"
        
        if os.path.exists(base):
            run = int(max(os.listdir(base)))
        # if this run hasnt been saved
        if not self.saved:
            run += 1
            
        return base + f"{run:02d}/"
    
    def get_step(self, step, steps):
        max_digits = int(np.log10(steps // 10000) + 1)
        return f"{step//10000:0{max_digits}d}"
    
    # model evaluation logging
    def log(self, row: dict):
        # Initialize lazily so columns match first row’s keys
        if self._writer is None:
            self.fieldnames = list(row.keys())
            # assert path exist or make it
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            # make csv
            self._f = open(self.path+"metrics.csv", "a", newline="")
            self._writer = csv.DictWriter(self._f, fieldnames=self.fieldnames)
            if self._f.tell() == 0:   # new file
                self._writer.writeheader()
        self._writer.writerow(row)
        self._f.flush()  # crash-safe
    
    # print info
    def print(self, step, reward, duration):
        print(f"Episode {step//1000}k:")
        print(f"AVG Reward: {int(reward)}")
        print(f"AVG Duration: {int(duration)}")
        print()
        
    # model saving
    def time(self, mode, step):
        if mode == "eval":
            e = 10000 # eval / 5000 steps
            return step % e == 0
        elif mode == "save":
            return step % 50000 == 0
        elif mode == "update":
            return step >= 10000
        print("invalid mode")

    def close(self):
        if hasattr(self, "_f"):
            self._f.close()