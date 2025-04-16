from matplotlib import ticker
import pandas as pd

#Load the dataframe from the CSV file
file_path = './float_mix_scaling_soft_N15_M14.csv'
df_scaling_soft = pd.read_csv(file_path)
df_scaling_soft['ratio makespan'] = (df_scaling_soft['ratio makespan'] - 1) * 100

#Load the dataframe from the CSV file
file_path = './float_mix_scaling_extreme_N15_M14.csv'
df_scaling_extreme = pd.read_csv(file_path)
df_scaling_extreme['ratio makespan'] = (df_scaling_extreme['ratio makespan'] - 1) * 100

#Load the dataframe from the CSV file
file_path = './float_wide_times_N15_M14.csv'
df_wide_times = pd.read_csv(file_path)
df_wide_times['ratio makespan'] = (df_wide_times['ratio makespan'] - 1) * 100


#Load the dataframe from the CSV file
file_path = './float_real_benchs_N15_M14.csv'
df_real_benchs = pd.read_csv(file_path)
df_real_benchs['ratio makespan'] = (df_real_benchs['ratio makespan'] - 1) * 100

import matplotlib.pyplot as plt

# Plot the 'step' column against the 'ratio makespan' column
plt.figure(figsize=(6.96, 5))
fig, ax = plt.subplots()
plt.plot(df_real_benchs['step'], df_real_benchs['ratio makespan'], linestyle='-', linewidth=2,label='Rodinia + Altis')
plt.plot(df_scaling_soft['step'], df_scaling_soft['ratio makespan'], linestyle='-', linewidth=2,label='MixScalingUniform')
plt.plot(df_scaling_extreme['step'], df_scaling_extreme['ratio makespan'], linestyle='-', linewidth=2,label='MixScalingExtreme')
plt.plot(df_wide_times['step'], df_wide_times['ratio makespan'], linestyle='-', linewidth=2,label='WideTimes')
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((6, 6))  # Fija la escala en 10^6

ax.xaxis.set_major_formatter(formatter)

plt.xlabel('Timesteps', fontsize=13)
plt.xticks(fontsize=13)
plt.ylabel(r'Max distance to optimum $p_{\text{opt}}$ (%)', fontsize=13)
plt.yticks(fontsize=13)
plt.ylim([79, 105])
plt.legend(fontsize=13)
plt.grid(axis='y', linestyle='--')

plt.tight_layout(pad=0)
plt.savefig('C:/Users/jorvi/Downloads/non_reconfiguration_curves.pdf', format='pdf')

plt.show()