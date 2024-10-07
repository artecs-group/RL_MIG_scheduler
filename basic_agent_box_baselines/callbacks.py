from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # obs_preprocessed = self.locals["obs_tensor"]
        # print(f"Step: {self.num_timesteps}, Preprocessed Observation: {obs_preprocessed}")
        return True