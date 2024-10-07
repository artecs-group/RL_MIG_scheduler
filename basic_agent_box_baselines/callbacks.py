from stable_baselines3.common.callbacks import BaseCallback
from env import SchedEnv
from utils import partition_map


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


def _makespan_lower_bound(dict_cont_times):
    for num_task, times in dict_cont_times.items():
        pass
        

class CustomCallback(BaseCallback):
    def __init__(self, M, N, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.M = M
        self.N = N

    def _on_step(self):
        env_test = SchedEnv({"N": self.N, "M": self.M})
        env_test.reset()
        done = False
        while not done:
            action, _ = self.model.predict(env_test.get_numpy_obs_state(), action_masks=env_test.valid_action_mask())
            _, _, done, _, _ = env_test.step(action)

        print(env_test.actions)
        print(_compute_makespan(env_test.init_state, env_test.actions))
        # obs_preprocessed = self.locals["obs_tensor"]
        # print(f"Step: {self.num_timesteps}, Preprocessed Observation: {obs_preprocessed}")
        return True