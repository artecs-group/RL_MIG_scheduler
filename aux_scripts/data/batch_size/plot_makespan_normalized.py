from matplotlib import ticker
import pandas as pd

#Load the dataframe from the CSV file
file_path = './ratios_N14_M14.csv'
df_N14 = pd.read_csv(file_path)
df_N14['ratio makespan'] = (df_N14['ratio makespan'] - 1) * 100

file_path = './ratios_N7_M14.csv'
df_N7 = pd.read_csv(file_path)
df_N7['ratio makespan'] = (df_N7['ratio makespan'] - 1) * 100

file_path = './ratios_N21_M14.csv'
df_N21 = pd.read_csv(file_path)
df_N21['ratio makespan'] = (df_N21['ratio makespan'] - 1) * 100

file_path = './ratios_N28_M14.csv'
df_N28 = pd.read_csv(file_path)
df_N28['ratio makespan'] = (df_N28['ratio makespan'] - 1) * 100

import matplotlib.pyplot as plt

# Plot the 'step' column against the 'ratio makespan' column
plt.figure(figsize=(6.96, 5))
fig, ax = plt.subplots()
plt.plot(df_N7['step'], df_N7['ratio makespan'], linestyle='-', linewidth=2,label='N=7')
plt.plot(df_N14['step'], df_N14['ratio makespan'], linestyle='-', linewidth=2,label='N=14')
plt.plot(df_N21['step'], df_N21['ratio makespan'], linestyle='-', linewidth=2,label='N=21')
plt.plot(df_N28['step'], df_N28['ratio makespan'], linestyle='-', linewidth=2,label='N=28')

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((6, 6))  # Fija la escala en 10^6

ax.xaxis.set_major_formatter(formatter)

plt.xlabel('Timesteps', fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(r'Max distance to optimum $p_{\text{opt}}$ (%)', fontsize=13)
plt.yticks(fontsize=13)
plt.ylim([0, 90])
plt.legend(title="Batch size N", fontsize=13, title_fontsize=13)
plt.grid(axis='y', linestyle='--')


plt.tight_layout(pad=0)
plt.savefig('C:/Users/jorvi/Downloads/batch_size.pdf', format='pdf')

plt.show()