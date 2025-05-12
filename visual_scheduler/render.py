from pprint import pprint
from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from FAR_scheduler.algorithm import *
from FAR_scheduler.plotting import draw_rects_tree
from utils import partition_map, action_to_str, compute_makespan
import copy


class Window:
    colors = plt.cm.tab20.colors

    def __init__(self, initial_env, lower_bound = None, mem_size = 5, model_trained = None):
        self._heuristic_solve(initial_env)
        self.model_trained = model_trained
        self.lower_bound = lower_bound
        figsize = (30, 5)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': [4, 6]})  
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2)
        # Crear los botones de "Siguiente" y "Anterior"
        axprev = plt.axes([0.3, 0, 0.1, 0.075])
        axnext = plt.axes([0.4, 0, 0.1, 0.075])
        axstep = plt.axes([0.5, 0, 0.1, 0.075])

        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self._previous_figure)
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self._next_figure)
        self.bstep = Button(axstep, 'Step' if self.model_trained else 'Random Step')
        self.bstep.on_clicked(self._step)

        self.mem_size = mem_size
        self.envs = [initial_env]
        self.current_env = 0
        self.terminated = False
        self.bprev.ax.set_visible(False)
        self.bnext.ax.set_visible(False)
        self._render_env(initial_env)
        plt.show()

    def _previous_figure(self, event):
        if self.current_env == 0:
            self.bprev.ax.set_visible(False)
            return
        self.current_env -= 1
        if self.current_env < len(self.envs) - 1:
            self.bnext.ax.set_visible(True)
            self.bstep.ax.set_visible(False)
        if self.current_env == 0:
            self.bprev.ax.set_visible(False)
        self._render_env(self.envs[self.current_env])

    def _next_figure(self, event):
        if self.current_env == len(self.envs) - 1:
            self.bnext.ax.set_visible(False)
            return
        self.current_env += 1
        if self.current_env > 0:
            self.bprev.ax.set_visible(True)
        if self.current_env == len(self.envs) - 1:
            self.bnext.ax.set_visible(False)
            self.bstep.ax.set_visible(True)
        self._render_env(self.envs[self.current_env])

    def _best_scaling_size(self, ready_tasks):
        pos_size = {0: 1, 1: 2, 2: 3, 3: 4, 4: 7}
        best_scaling = []
        for task in ready_tasks:
            if task[-1] == 0:
                return best_scaling
            best_scaling.append(1)
            min_a = float("inf")
            for i, time in enumerate(task[:5]):
                size = pos_size[i]
                a = size * time
                if a < min_a:
                    min_a = a
                    best_scaling[-1] = size
        return best_scaling
            
            
    def _heuristic_solve(self, env):
        times = [time for list_times in env.dic_cont_times.values() for time in list_times]
        sizes = [1,2,3,4,7]
        times = [[(i, sizes[j], time) for j, time in enumerate(time_task)] for i, time_task in enumerate(times)]
        allotmets_family = create_allotments_family(times, 7)
        lb_makespane_opt = lower_bound_makespan_opt(allotmets_family, 7)
        tree = moldable_scheduler_tree(7, allotmets_family, "A100")
        refinement(tree, "A100")
        draw_rects_tree(7, tree, lb_makespane_opt)
        
    

    def _step(self, event, action = None):
        if self.terminated or self.current_env != len(self.envs) -1:
            self.bstep.ax.set_visible(False)
            return
        env = copy.deepcopy(self.envs[-1])
        if self.model_trained:
            action, _ = self.model_trained.predict(env.get_numpy_obs_state(), action_masks=env.valid_action_mask())
        else:
            action = np.random.choice(np.flatnonzero(env.obs["action_mask"]))
        obs, reward, terminated, _, _ = env.step(action)
        print("Terminated", terminated, "Reward", reward)
        self.terminated = terminated
        self._render_env(env)
        if len(self.envs) < self.mem_size:
            self.current_env += 1
        else:
            self.envs.pop(0)

        self.envs.append(env)
        self.bprev.ax.set_visible(True)
            
    def _clear_axes(self):
        self.ax1.clear()
        self.ax2.clear()

    def _render_env(self, env):
        self._clear_axes()
        partition = env.obs["partition"]
        slices_t = env.obs["slices_t"]
        ready_tasks = env.obs["ready_tasks"]
        action_mask = env.obs["action_mask"]
        title = (f"Initial state." if env.last_action is None else f"Last action: {action_to_str(env.last_action)}.") + f" Reward: {env.acum_reward:.2f}."
        # title += f" Real time lower bound: {self.lower_bound:.2f}" if self.lower_bound else ""
        if self.terminated:
            if self.lower_bound:
                makespan = compute_makespan(env.init_state, env.actions)
                ratio = makespan / self.lower_bound
                title = f"Real lower bound: {self.lower_bound:.2f}.  Real Makespan: {makespan:.2f}. Ratio: {ratio:.2f}."
        
        self.fig.suptitle(title)
        slice_i = 0
        sizes = partition_map[partition]["sizes"]
        if sizes[0] == 3:
            sizes[0] = 4
        for instance_size in sizes:
            # Representa los 7 valores de state["slices_t"] en una gráfica de barras
            rect = patches.Rectangle((slice_i, 0), instance_size, slices_t[slice_i], alpha = 0.55,\
                                            linewidth = 1, facecolor = self.colors[env.num_task_slices[slice_i] % len(self.colors)], edgecolor = 'black')
            self.ax1.add_patch(rect)
            slice_i += instance_size
        self.ax1.set_ylim([0, env.M])
        self.ax1.set_xlim([0, 7])
        self.ax1.set_xlabel("Slices")
        self.ax1.set_ylabel("Time")
        self.ax1.set_yticks(range(0, env.M+1))
        self.ax1.set_title(f"GPU Config. {partition_map[partition]['sizes']}")

        slices_l= [1,2,3,4,7]
        # Create a bar chart for each task in state["ready_tasks"]
        
        for i, task in enumerate(ready_tasks):
            if task[-1] == 0:
                continue
            # Calculate the x positions for the bars
            x = [j + i * 5 for j in range(5)]
            # Create the bar chart
            self.ax2.bar(x, task[:5], color=self.colors[env.num_type_task[i] % len(self.colors)], alpha=0.55)
            for j, slice_pos in enumerate(x):
                self.ax2.text(slice_pos, -env.M/100, slices_l[j], ha='center', va='top')
            self.ax2.text(2 + i * 5, -env.M/20, str(f"{task[-1]} {'task'if task[-1] == 1 else 'tasks'}"), ha='center', va='top')
            
        # Set y labels for the bar chart
        self.ax2.set_ylabel("Time")
        self.ax1.set_ylim([0, env.M])
        # Set the title for the bar chart
        self.ax2.set_title("Ready tasks")
        # Remove the ticks on the x-axis of ax2
        self.ax2.set_xticks([])
        self.ax2.set_yticks(range(0, env.M+1))

        # Add text below the figures
        wait, reconfigure, n_task = action_mask[0], action_mask[1:17], action_mask[17:]


        plt.draw()
        