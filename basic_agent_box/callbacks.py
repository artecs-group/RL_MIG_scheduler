from ray.rllib.algorithms.callbacks import DefaultCallbacks
import os
if os.getcwd().split('/')[-1] != "basic_agent_box" and os.getcwd().split('\\')[-1] != "basic_agent_box":
    os.chdir("./basic_agent_box")
from render import Window

class CustomMetricsCallback(DefaultCallbacks):
    def on_postprocess_trajectory(self, *, worker, episode, agent_id, policy_id, policies, postprocessed_batch, original_batches, **kwargs):
        return
        # Code for see outputs per inference during training
        # Obtener las observaciones y logits del lote procesado
        observations = postprocessed_batch["obs"]  # Observaciones
        actions = postprocessed_batch["actions"]  # Acciones tomadas
        logits = postprocessed_batch["action_dist_inputs"]  # Logits de salida

        # Imprimir las salidas para cada paso del lote
        for i in range(len(observations)):
            print(f"Paso {i + 1}:")
            print(f"  Observación: {observations[i]}")
            print(f"  Acción tomada: {actions[i]}")
            print(f"  Logits: {logits[i]}")
        input("Presiona Enter para continuar...")