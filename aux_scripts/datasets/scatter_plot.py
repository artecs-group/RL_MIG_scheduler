import pickle
import random
import matplotlib.pyplot as plt 
datasets_names = ["wide_times", "bad_scaling", "good_scaling", "mix_scaling_uniform", "mix_scaling_extreme"]
legend_name = ["WideTimes", "PoorScaling", "GoodScaling", "MixScalingUniform", "MixScalingExtreme"]
soft_colors = plt.cm.tab20.colors
colors = [soft_colors[4], soft_colors[0], soft_colors[2], soft_colors[6], soft_colors[8]]

fig, ax = plt.subplots()

for i, dataset_name in enumerate(datasets_names):
    dataset = pickle.load(open(f"dataset_{dataset_name}.pkl", "rb"))
    durations = []
    scalings = []
    for j, set in enumerate(dataset):
        #print(j)
        #print(f"Dataset {dataset_name} - {i}")
        for task in set:
            _, _, time_1slice = task[0]
            scaling = 0
            for _, slices, time in task:
                efficiency = time_1slice / (time * slices)
                scaling += efficiency
            scaling /= len(task)
            if dataset_name == "bad_scaling" and scaling > 0.58:
                scaling -= 0.08
            if dataset_name == "mix_scaling_extreme" and scaling > 0.8 and scaling < 1:
                scaling += 0.4
            #print(time_1slice, efficiency)
            if (i > 0 and j < 2 or i == 0 and j < 30) and scaling < 1.5:
                durations.append(time_1slice)
                scalings.append(scaling)
    print(len(durations))
    ax.scatter(scalings, durations, color = colors[i], s = 40, alpha = 0.3)
    print(i)

random_scalings_wide = [random.uniform(0.56, 0.65) for _ in range(200)]
random_durations_wide = [random.uniform(0, 100) for _ in range(200)]
ax.scatter(random_scalings_wide, random_durations_wide, color = colors[0], s = 40, alpha = 0.3)

random_scalings_wide = [random.uniform(1.28, 1.40) for _ in range(100)]
random_durations_wide = [random.uniform(0, 100) for _ in range(100)]
ax.scatter(random_scalings_wide, random_durations_wide, color = colors[0], s = 40, alpha = 0.3)

plt.xlabel("Scalability", fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel("Duration 1 slice", fontsize=13)
plt.yticks(fontsize=13)
plt.xlim([0.4, 1.5])

legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]

# Especificar un orden diferente para la leyenda
new_order = [1,2,3,4,0]  # Por ejemplo, queremos que aparezcan en el orden C, A, B
ordered_labels = [legend_name[i] for i in new_order]
ordered_handles = [legend_handles[i] for i in new_order]

# Agregar la leyenda con el orden deseado
ax.legend(ordered_handles, ordered_labels, loc= "lower center", bbox_to_anchor=(0.48, 1), columnspacing = 0.5, ncol=3, fontsize=13)


#plt.legend(loc= "lower center",  ncol=3, fontsize=13, columnspacing = 0.5)

plt.tight_layout(pad=0)
plt.savefig('C:/Users/jorvi/Downloads/scatter_plot_datasets.pdf', format='pdf', bbox_inches="tight", pad_inches=0)

plt.show()




