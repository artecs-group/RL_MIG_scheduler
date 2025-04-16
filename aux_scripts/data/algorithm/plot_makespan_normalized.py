from matplotlib import ticker
import pandas as pd

#Load the dataframe from the CSV file
file_path = './DQN_64x64_N14_M14.csv'
df_DQN64 = pd.read_csv(file_path)
df_DQN64['ratio makespan'] = (df_DQN64['ratio makespan'] - 1) * 100

file_path = './DQN_256x256_N14_M14.csv'
df_DQN256 = pd.read_csv(file_path)
df_DQN256['ratio makespan'] = (df_DQN256['ratio makespan']- 1) * 100

file_path = './PPO_64x64_N14_M14.csv'
df_PPO64 = pd.read_csv(file_path)
df_PPO64['ratio makespan'] = (df_PPO64['ratio makespan']- 1) * 100

file_path = './PPO_256x256_N14_M14.csv'
df_PPO256 = pd.read_csv(file_path)
df_PPO256['ratio makespan'] = (df_PPO256['ratio makespan']- 1) * 100


import matplotlib.pyplot as plt

# Plot the 'step' column against the 'ratio makespan' column
plt.figure(figsize=(6.96, 5))
fig, ax = plt.subplots()
plt.plot(df_DQN64['step'], df_DQN64['ratio makespan'], linestyle='-', linewidth=2,label='DQN, MLP 64x64')
plt.plot(df_PPO64['step'], df_PPO64['ratio makespan'], linestyle='-', linewidth=2, label='PPO, MLP 64x64')
plt.plot(df_DQN256['step'], df_DQN256['ratio makespan'], linestyle='-', linewidth=2, label='DQN, MLP 256x256')
plt.plot(df_PPO256['step'], df_PPO256['ratio makespan'], linestyle='-', linewidth=2, label='PPO, MLP 256x256')

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((6, 6))  # Fija la escala en 10^6

ax.xaxis.set_major_formatter(formatter)

plt.xlabel('Timesteps', fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(r'Max distance to optimum $p_{\text{opt}}$ (%)', fontsize=13)
plt.yticks(fontsize=13)
plt.ylim([0, 90])
plt.legend(fontsize=13)
plt.grid(axis='y', linestyle='--')


plt.tight_layout(pad=0)
plt.savefig('C:/Users/jorvi/Downloads/training_algorithm.pdf', format='pdf')

plt.show()