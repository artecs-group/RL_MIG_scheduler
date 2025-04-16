import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sé que es un poco chapucero, pero sólo quiero la gráfica no un código legible y bonito

df = pd.read_csv("gaussian_A100_1g.csv")
df["mean_1g"] = df[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df = df[["prueba", "mean_1g"] ]
df["speed_1g"] = df["mean_1g"] / df["mean_1g"]
df_2g = pd.read_csv("gaussian_A100_2g.csv")
df["mean_2g"] = df_2g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df["speed_2g"] = df["mean_1g"] / df["mean_2g"]
df_3g = pd.read_csv("gaussian_A100_3g.csv")
df["mean_3g"] = df_3g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df["speed_3g"] = df["mean_1g"] / df["mean_3g"]
df_4g = pd.read_csv("gaussian_A100_4g.csv")
df["mean_4g"] = df_4g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df["speed_4g"] = df["mean_1g"] / df["mean_4g"]
df_7g = pd.read_csv("gaussian_A100_7g.csv")
df["mean_7g"] = df_7g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df["speed_7g"] = df["mean_1g"] / df["mean_7g"]

print(df)


time_cols = ["mean_1g", "mean_2g", "mean_3g", "mean_4g", "mean_7g"]
speed_cols = ["speed_1g", "speed_2g", "speed_3g", "speed_4g", "speed_7g"]
gaussian_times_4096 = df[df["prueba"] == "gaussian_4096"][time_cols].iloc[0].tolist()
gaussian_speeds_4096 = df[df["prueba"] == "gaussian_4096"][speed_cols].iloc[0].tolist()
gaussian_times_1024 = df[df["prueba"] == "gaussian_1024"][time_cols].iloc[0].tolist()
gaussian_speeds_1024 = df[df["prueba"] == "gaussian_1024"][speed_cols].iloc[0].tolist()
GPCs = np.array([1, 2, 3, 4, 7])
index = GPCs

bar_width = 0.35

plt.rcParams["figure.figsize"] = (6.5, 3)

for i in range(len(index)):
    plt.text(index[i] -  bar_width/2, gaussian_speeds_1024[i] / 2, f"{gaussian_times_1024[i]:.2f}s", ha='center', va='center', color='black', fontsize=11)

plt.bar(index-bar_width/2, gaussian_speeds_1024, bar_width, label='Input size 1024', alpha = 0.75)

for i in range(len(index)):
    plt.text(index[i] + bar_width/2, gaussian_speeds_4096[i] / 2, f"{gaussian_times_4096[i]:.1f}s", ha='center', va='center', color='black', fontsize=11)

plt.bar(index + bar_width/2, gaussian_speeds_4096, bar_width, label='Input size 4096', alpha = 0.75)

plt.xticks(np.array([1, 2, 3, 4, 5, 6, 7]), np.array([1, 2, 3, 4, 5, 6, 7]), fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Number of slices", fontsize=13)
plt.ylabel("Speedup over 1 slice", labelpad=5, fontsize=13)

legend1 = plt.legend(loc='upper left', fontsize=13)
plt.gca().add_artist(legend1)
x = np.linspace(1,7,100)
linear_scaling, = plt.plot(x, x, color="black", linestyle='--', label= "Linear scaling")
plt.legend(fontsize=13, handles=[linear_scaling])


plt.xlim(0.5, 7.5)
plt.ylim(0, 3.5)
plt.grid(axis='y', linestyle='--')

plt.tight_layout(pad=0)
#plt.savefig("C:/Users/jorvi/Downloads/gaussian_MIG_scaling1.pdf")

plt.show()
