from gymnasium.spaces import Box

from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN

torch, nn = try_import_torch()


class TorchParametricActionsModel(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        action_embed_size,
        **kw
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )

        self.action_embed_model = TorchFC(
            obs_space,
            action_space,
            action_embed_size,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the predicted action embedding
        action_embed, _ = self.action_embed_model({"obs": input_dict["obs"]["observations"]})

        # Expand the model output to [BATCH, 1, EMBED_SIZE]. Note that the
        # avail actions tensor is of shape [BATCH, MAX_ACTIONS, EMBED_SIZE].
        intent_vector = torch.unsqueeze(action_embed, 1)

        # Batch dot product => shape of logits is [BATCH, MAX_ACTIONS].
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)

        # Mask out invalid actions (use -inf to tag invalid).
        # These are then recognized by the EpsilonGreedy exploration component
        # as invalid actions that are not to be chosen.
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)

        return action_logits + inf_mask, state

    def value_function(self):
        return self.action_embed_model.value_function()

