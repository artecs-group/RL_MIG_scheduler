from matplotlib import ticker
import pandas as pd

#Load the dataframe from the CSV file
file_path = './ratios_N14.csv'
df = pd.read_csv(file_path)
df['ratio makespan'] = (df['ratio makespan'] - 1) * 100

file_path = './ratios_N14_M28.csv'
df_M28 = pd.read_csv(file_path)
df_M28['ratio makespan'] = (df_M28['ratio makespan']- 1) * 100

file_path = './ratios_N14_M21.csv'
df_M21 = pd.read_csv(file_path)
df_M21['ratio makespan'] = (df_M21['ratio makespan']- 1) * 100

file_path = './ratios_N14_M14.csv'
df_M14 = pd.read_csv(file_path)
df_M14['ratio makespan'] = (df_M14['ratio makespan']- 1) * 100

file_path = './ratios_N14_M7.csv'
df_M7 = pd.read_csv(file_path)
df_M7['ratio makespan'] = (df_M7['ratio makespan']- 1) * 100



import matplotlib.pyplot as plt

# Plot the 'step' column against the 'ratio makespan' column
plt.figure(figsize=(6.96, 5))
fig, ax = plt.subplots()
plt.plot(df['step'], df['ratio makespan'], linestyle='-', linewidth=2,label='Continuous (original)')
plt.plot(df_M28['step'], df_M28['ratio makespan'], linestyle='-', linewidth=2, label='Discrete M=28')
plt.plot(df_M21['step'], df_M21['ratio makespan'], linestyle='-', linewidth=2, label='Discrete M=21')
plt.plot(df_M14['step'], df_M14['ratio makespan'], linestyle='-', linewidth=2, label='Discrete M=14')
plt.plot(df_M7['step'], df_M7['ratio makespan'], linestyle='-', linewidth=2, label='Discrete M=7')

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
plt.savefig('C:/Users/jorvi/Downloads/discretization_levels.pdf', format='pdf')

plt.show()