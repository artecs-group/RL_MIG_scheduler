from collections import OrderedDict
import gymnasium as gym
import ray
import numpy as np
from ray.rllib.algorithms import ppo
from gymnasium.spaces import Dict, Discrete, MultiDiscrete, MultiBinary
from utils import partition_map, instance_size_map, canonical_sort_tasks, print_obs



class SchedEnv(gym.Env):
    def __init__(self, env_config):
        self.N = env_config["N"]
        self.M = env_config["M"]
        self.reconfig_time = 1 # Tiempo de reconfiguración 1 unidad, pensar cómo modelar adecuadamente
        self.observation_space = Dict({
            "observations": Dict({
                "partition": Discrete(19), # 19 particiones posibles
                "slices_t": MultiDiscrete([(self.M + 1)] * 7), # Cuanto le queda (de 0 a M) a cada slice
                "ready_tasks": MultiDiscrete([[(self.M + 1)] * 5 + [(self.N + 1)]] * self.N) # El tipo de tarea (tiempo que tarda con cada una de los 5 tamaños e instancia) de las hasta N tareas libres, y la cantidad de tareas que hay de ese tipo (componente 6)
            }),
            "action_mask": MultiBinary(1 + 19 + 7 * self.N,) # Máscara de acciones válidas
        })

        self.action_space = Discrete(1 + 19 + 7 * self.N) # 1 accion esperar, 19 acciones de configuración, y 7*N acciones de asignar tarea


    def _get_action_mask(self):
        current_partition = self.obs["observations"]["partition"]
        slices_t = self.obs["observations"]["slices_t"]
        ready_tasks = self.obs["observations"]["ready_tasks"]

        # La acción de esperar sólo es válida si hay tareas en la GPU
        wait = 1 if any(slice_t != 0 for slice_t in slices_t) else 0


        # Reconfiguraciones válidas
        reconfig_mask = [1] * 19
        # Prohibido reconfigurar a la partición actual
        reconfig_mask[current_partition - 1] = 0
        # Prohibido reconfigurar a particiones en que haya que cambiar slices en uso
        for future_partition in range(1, 20):
            for curr_slice in range(7):
                if partition_map[current_partition]["slices"][curr_slice] != partition_map[future_partition]["slices"][curr_slice] and slices_t[curr_slice] > 0:
                    reconfig_mask[future_partition - 1] = 0

        # Acciones válidas sobre tareas ready
        select_ready_task = [1] * 7 * self.N
        # Prohibido coger una tarea para ejecutar en la GPU más allá de los tipos disponibles
        for i, task in enumerate(ready_tasks):
            n_availables = task[-1]
            # Si en ready hay una no disponible todas las de detrás no están disponibles por el orden canónico
            if n_availables == 0:
                for j in range(i*7, 7 * self.N):
                    select_ready_task[j] = 0
                break
        
        # Prohibido asignar la tarea ready a una instancia con indice superior a la cantidad de instancias de la partición actual
        first_forbidden_instance = len(partition_map[current_partition]["sizes"])
        for task in range(self.N):
            for instance in range(first_forbidden_instance, 7):
                select_ready_task[task * 7 + instance] = 0

        first_instance_slice = 0
        # Prohibido asignar la tarea ready a una instancia con índice <= cantidad de instancias de la partición actual
        for instance_index in range(first_forbidden_instance):
            if slices_t[first_instance_slice] > 0:
                for task in range(self.N):
                    select_ready_task[task * 7 + instance_index] = 0
            first_instance_slice += partition_map[current_partition]["sizes"][instance_index]

        return [wait] + reconfig_mask + select_ready_task
        


    def _get_numpy_obs_state(self):
        return OrderedDict([('action_mask', np.array(self.obs["action_mask"])),
                            ('observations', OrderedDict([('partition', np.int64(self.obs["observations"]["partition"])),
                                                          ('ready_tasks', np.array(self.obs["observations"]["ready_tasks"])),
                                                          ('slices_t', np.array(self.obs["observations"]["slices_t"]))]))])



    def reset(self, seed = None, options = None):

        num_ready = self.N if np.random.rand() < 0.8 else np.random.randint(1, self.N)
        pending_tasks = [sorted(np.random.randint(1, self.M + 1, size=5), reverse=True) for _ in range(num_ready)]
        init_partition = 1 # Por ahora, consideramos que se empieza en la partición 1 (una sola instancia de tamaño 7 siempre)
        init_slice_t = [0] * 7 # Consideramos que todos los slices están libres al principio
        ready_tasks_canonical = canonical_sort_tasks(self.M, pending_tasks[:self.N]) # Las siguientes N tareas pendientes, se ordenan canónicamente y colocando como 6ª componente la cantidad de veces que se repite
        # Usamos el 0 para rellenar posiciones vacías en la representación de tareas ready
        init_ready_tasks = ready_tasks_canonical + [[0] * 6] * (self.N - len(ready_tasks_canonical)) # Rellenamos con arrays de 6 ceros hasta N

        self.obs = {
            "observations":{
                "partition": init_partition,
                "slices_t": init_slice_t,
                "ready_tasks": ready_tasks_canonical
            }
        }
        self.obs["action_mask"] = self._get_action_mask()

        
        return self._get_numpy_obs_state(), {}
    

    def render(self):
        print_obs(self.obs)



    def _is_terminated(self):
        # Si todas las tareas están terminadas, y todo lo que hay en la GPU ha acabado, el episodio termina
        for slice_t in self.obs["observations"]["slices_t"]:
            if slice_t > 0:
                return False

        for ready_task in self.obs["observations"]["ready_tasks"]:
            if ready_task[-1] > 0:
                return False
        
        return True

    def step(self, action):
        # Esperar
        if action == 0:
            pass
        # Reconfigurar
        elif action <= 19:
            self.obs["observations"]["partition"] = action
            # Para la reconfiguración introduzco una tarea ficticia en las instancias libres
            for slice, time in enumerate(self.obs["observations"]["slices_t"]):
                if time == 0:
                    self.obs["observations"]["slices_t"][slice] = self.reconfig_time
        # Asignar tarea
        else:
            task = (action - 20) // 7
            instance = (action - 20) % 7
            # Quitamos la tarea de las ready_tasks
            self.obs["observations"]["ready_tasks"][task][-1] -= 1
            # Si es la última de cierto tipo, quitamos el tipo de ready task
            if self.obs["observations"]["ready_tasks"][task][-1] == 0:
                self.obs["observations"]["ready_tasks"].pop(task)
                # Añadimos un tipo de tarea vacío al final para mantener la dimensionalidad
                self.obs["observations"]["ready_tasks"].append([0] * 6)
            # Aumentamos el tiempo que tarda la tarea para el tamaño de la instancia en todos los slices de la instancia
            instance_size = partition_map["sizes"][instance]
            task_time = self.obs["observations"]["ready_tasks"][task][instance_size_map[instance_size]]    
            for i, instance_slice in enumerate(partition_map["instances"]):
                if instance_slice == instance:
                    self.obs["observations"]["slices_t"][i] = task_time

        # Transitamos al primer slice que se libere
        min_slice_time = min(self.obs["observations"]["slices_t"])
        self.obs["observations"]["slices_t"] = [slice_time - min_slice_time for slice_time in self.obs["observations"]["slices_t"]]

        # Actualizamos la máscara de acciones para el nuevo estado
        self.obs["action_mask"] = self._get_action_mask()

        # Recompensamos con -tiempo transcurrido, para minimizar el makespan
        reward = -min_slice_time
        # Comprobamos si el episodio ha terminado
        terminated = self._is_terminated()
        
        truncated, info = False, {}

        return self._get_numpy_obs_state(), reward, terminated, truncated, info

    def close():
        pass




env_example = SchedEnv({"N": 5, "M": 3})
print("Example")
print(env_example.observation_space.sample())
initial_obs, _ = env_example.reset(seed=0)
env_example.render()
print("\n\nAfter waiting")
env_example.step(1)
env_example.render()