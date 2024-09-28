from matplotlib import patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
from utils import partition_map, _action_to_str
import copy


class Window:
    colors = plt.cm.tab20.colors

    def __init__(self, initial_env, mem_size = 5):
        figsize = (30, 5)
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=figsize, gridspec_kw={'width_ratios': [3, 3, 1]})  
        self.fig.set_size_inches(12, 5)
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.2)
        # Crear los botones de "Siguiente" y "Anterior"
        axprev = plt.axes([0.3, 0, 0.1, 0.075])
        axnext = plt.axes([0.4, 0, 0.1, 0.075])
        axstep = plt.axes([0.5, 0, 0.1, 0.075])

        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self._previous_figure)
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self._next_figure)
        self.bstep = Button(axstep, 'Random Step')
        self.bstep.on_clicked(self._step)

        self.mem_size = mem_size
        self.envs = [initial_env]
        self.current_env = 0
        self.bprev.ax.set_visible(False)
        self.bnext.ax.set_visible(False)
        self._render_env(initial_env)
        self.terminated = False
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
        

    def _step(self, event):
        if self.current_env != len(self.envs) -1:
            self.bstep.ax.set_visible(False)
            return
        env = copy.deepcopy(self.envs[-1])
        action = np.random.choice(np.flatnonzero(env.obs["action_mask"]))
        _, _, terminated, _, _= env.step(action)
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
        self.ax3.clear()

    def _render_env(self, env):
        self._clear_axes()
        state = env.obs["observations"]
        action_mask = env.obs["action_mask"]

        self.fig.suptitle((f"Initial state." if env.last_action is None else f"Last action: {_action_to_str(env.last_action)}.") + f" Reward: {env.acum_reward}")
        
        slice_i = 0
        for instance_size in partition_map[state["partition"]]["sizes"]:
            # Representa los 7 valores de state["slices_t"] en una gráfica de barras
            rect = patches.Rectangle((slice_i, 0), instance_size, state["slices_t"][slice_i], alpha = 0.55,\
                                            linewidth = 1, facecolor = self.colors[env.num_task_slices[slice_i] % len(self.colors)], edgecolor = 'black')
            self.ax1.add_patch(rect)
            slice_i += instance_size
        self.ax1.set_ylim([0, env.M])
        self.ax1.set_xlim([0, 7])
        self.ax1.set_xlabel("Slices")
        self.ax1.set_ylabel("Time")
        self.ax1.set_yticks(range(0, env.M+1))
        self.ax1.set_title(f"GPU Partition {partition_map[state['partition']]['sizes']}")

        slices_l= [1,2,3,4,7]
        # Create a bar chart for each task in state["ready_tasks"]
        
        for i, task in enumerate(state["ready_tasks"]):
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
        wait, reconfigure, n_task = action_mask[0], action_mask[1:20], action_mask[20:]

        # Remove axes and labels from ax3
        self.ax3.axis('off')

        self.ax3.set_title("Posible actions")
        # Add text to ax3
        self.ax3.text(0.5, 0.95, "Wait" if wait else "", ha='center', va='center', fontsize=12)

        self.ax3.text(0, 0.9, "Reconfigs:", ha='left', va='center', fontsize=12)
        height = 0.85
        for part, reconfig in enumerate(reconfigure):
            if reconfig == 1:
                self.ax3.text(0.5, height, str(partition_map[part+1]["sizes"]), ha='center', va='center', fontsize=12)
                height -= 0.05

        # # Draw rectangles on ax3
        # rect1 = patches.Rectangle((0.1, 0.1), 0.3, 0.3, linewidth=1, edgecolor='black', facecolor='red')
        # rect2 = patches.Rectangle((0.6, 0.1), 0.3, 0.3, linewidth=1, edgecolor='black', facecolor='blue')
        # ax3.add_patch(rect1)
        # ax3.add_patch(rect2)
        plt.draw()
        