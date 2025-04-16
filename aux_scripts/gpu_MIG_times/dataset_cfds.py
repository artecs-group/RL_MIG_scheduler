import pickle
import pandas as pd
import matplotlib.pyplot as plt

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

cfd_times = [df_1g["mean"][df_1g["prueba"] == "CFD"].values[0], df_2g["mean"][df_2g["prueba"] == "cfd"].values[0], df_3g["mean"][df_3g["prueba"] == "cfd"].values[0], df_4g["mean"][df_4g["prueba"] == "cfd"].values[0], df_7g["mean"][df_7g["prueba"] == "cfd"].values[0]]
cfd_times = [cfd_times]*20

with open(f"dataset_cfds.pkl", "wb") as f:
    pickle.dump(cfd_times, f)

