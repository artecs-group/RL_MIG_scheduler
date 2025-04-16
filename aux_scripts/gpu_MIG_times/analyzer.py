import pandas as pd
import matplotlib.pyplot as plt

# Sé que es un poco chapucero, pero sólo quiero la gráfica no un código legible y bonito

df_1g = pd.read_csv("1g_A100.csv")
df_1g["mean"] = df_1g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df_1g = df_1g[["prueba", "mean"]]
df_1g["speed"] = df_1g["mean"] / df_1g["mean"]

df_2g = pd.read_csv("2g_A100.csv")
df_2g["mean"] = df_2g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df_2g = df_2g[["prueba", "mean"]]
df_2g["speed"] = df_1g["mean"] / df_2g["mean"]

df_3g = pd.read_csv("3g_A100.csv")
df_3g["mean"] = df_3g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df_3g = df_3g[["prueba", "mean"]]
df_3g["speed"] = df_1g["mean"] / df_3g["mean"]

df_4g = pd.read_csv("4g_A100.csv")
df_4g["mean"] = df_4g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df_4g = df_4g[["prueba", "mean"]]
df_4g["speed"] = df_1g["mean"] / df_4g["mean"]

df_7g = pd.read_csv("7g_A100.csv")
df_7g["mean"] = df_7g[["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"]].mean(axis=1)
df_7g = df_7g[["prueba", "mean"]]
df_7g["speed"] = df_1g["mean"] / df_7g["mean"]

slices=[1,2,3,4,7]

#plt.rcParams["figure.figsize"] = (6.5, 2.5)
colors = plt.cm.tab20.colors
for i, (prueba, speeds) in enumerate(zip(df_1g["prueba"], zip(df_2g["speed"], df_3g["speed"], df_4g["speed"], df_7g["speed"]))):
    #if prueba == "pathfinder" or prueba == "nn" or prueba == "srad/srad_v2" or prueba == "heartwall" or prueba == "backprop"\
    #or prueba == "nw" or prueba == "lud":
    #    continue
    speeds = list(speeds)
    plt.plot(slices[1:], speeds, label = prueba, marker='o', color=colors[i % len(colors)])

plt.xlabel("Number of slices", fontsize=12)
plt.xticks(fontsize=12)
plt.ylabel("Speedup over 1 slice", labelpad=5, fontsize=12)
plt.yticks(fontsize=12)
plt.grid(axis='y', linestyle='--')

# Obtener la figura actual
fig = plt.gcf()
# Cambiar dimensiones (ancho, alto) en pulgadas
fig.set_size_inches(6.96, 5)

legend1 = plt.legend(loc='upper left', ncol=2, fontsize=12)
plt.gca().add_artist(legend1)
linear_scaling, = plt.plot(slices[1:], slices[1:], linestyle='--', label= "Linear scaling", color="black")
plt.legend(["Linear scaling"], fontsize=12, handles=[linear_scaling])
plt.tight_layout(pad=0)
plt.savefig("C:/Users/jorvi/Downloads/MIG_scaling.pdf")
plt.show()
