from stable_baselines3.common.callbacks import BaseCallback
from env import SchedEnv
from utils import partition_map
import csv



def _makespan_lower_bound(ready_tasks):
    pos_to_slices = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7}
    sum_min_area = 0
    for times_t in ready_tasks:
        areas = [pos_to_slices[pos] * time for pos, time in enumerate(times_t)]
        min_area = min(areas)
        sum_min_area += min_area
    return sum_min_area / 7

        

class CustomCallback(BaseCallback):
    def __init__(self, N, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.N = N

    def calculate_ratio(self, N, model):
        env_test = SchedEnv({"N": N})
        sum_ratios = 0
        num_episodes = 150
        for _ in range(num_episodes):
            env_test.reset()
            lower_bound = _makespan_lower_bound(env_test.obs["ready_tasks"])
            done = False
            while not done:
                action, _ = model.predict(env_test.get_numpy_obs_state(), action_masks=env_test.valid_action_mask())
                _, _, done, _, _ = env_test.step(action)

            makespan = -env_test.acum_reward
            ratio = makespan / lower_bound
            sum_ratios += ratio
            # obs_preprocessed = self.locals["obs_tensor"]
            # print(f"Step: {self.num_timesteps}, Preprocessed Observation: {obs_preprocessed}")
        
        with open(f'ratios_N{N}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model.num_timesteps, sum_ratios / num_episodes])
        

    def _on_step(self):      
        self.calculate_ratio(self.N, self.model)
        return True