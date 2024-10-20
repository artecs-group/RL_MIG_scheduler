from stable_baselines3.common.callbacks import BaseCallback
from env import SchedEnv
from utils import partition_map
import csv


def _compute_makespan(init_state, actions):
    partition = init_state["partition"]
    slices_t = init_state["slices_t"]
    for type, val in actions:
        if type == "reconfig":
            partition = val
        elif type == "assign":
            time, instance = val
            for pos, slice_ins in enumerate(partition_map[partition]["instances"]):
                if slice_ins == instance:
                    slices_t[pos] += time
    return max(slices_t)


def _makespan_lower_bound(dic_cont_times):
    pos_to_slices = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7}
    sum_min_area = 0
    for num_task, times in dic_cont_times.items():
        for times_t in times:
            areas = [pos_to_slices[pos] * time for pos, time in enumerate(times_t)]
            min_area = min(areas)
            sum_min_area += min_area
    return sum_min_area / 7     

class CustomCallback(BaseCallback):
    def __init__(self, M, N, type_tasks = "good_scaling", verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.M = M
        self.N = N
        self.type_tasks = type_tasks

    def calculate_ratio(self, N, M, model):
        env_test = SchedEnv({"N": N, "M": M})
        sum_ratios = 0
        num_episodes = 150
        for _ in range(num_episodes):
            env_test.reset()
            lower_bound = _makespan_lower_bound(env_test.dic_cont_times)
            done = False
            while not done:
                action, _ = model.predict(env_test.get_numpy_obs_state(), action_masks=env_test.valid_action_mask())
                _, _, done, _, _ = env_test.step(action)

            makespan = _compute_makespan(env_test.init_state, env_test.actions)
            ratio = makespan / lower_bound
            sum_ratios += ratio
            # obs_preprocessed = self.locals["obs_tensor"]
            # print(f"Step: {self.num_timesteps}, Preprocessed Observation: {obs_preprocessed}")
        
        with open(f'ratios_N{N}_M{M}_{self.type_tasks}.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model.num_timesteps, sum_ratios / num_episodes])
        

    def _on_step(self):
        print("Num states visited", len(CallbackNumStates.visited_states))
        self.calculate_ratio(self.N, self.M, self.model)
        return True
    
class CallbackNumStates(BaseCallback):
    visited_states = set() # class variable
    def __init__(self, verbose=1):
        super(CallbackNumStates, self).__init__(verbose)

    def _on_step(self):
        self.visited_states.add(tuple(self.locals["obs_tensor"]))
        return True