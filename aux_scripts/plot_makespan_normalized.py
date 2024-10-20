import pandas as pd

#Load the dataframe from the CSV file
file_path = './data/ratios_N15.csv'
df = pd.read_csv(file_path)

file_path = './data/ratios_N15_M28.csv'
df_M28 = pd.read_csv(file_path)

file_path = './data/ratios_N15_M21.csv'
df_M21 = pd.read_csv(file_path)

file_path = './data/ratios_N15_M14.csv'
df_M14 = pd.read_csv(file_path)

file_path = './data/ratios_N15_M7.csv'
df_M7 = pd.read_csv(file_path)  

import matplotlib.pyplot as plt

# Plot the 'step' column against the 'ratio makespan' column
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['ratio makespan'],linestyle='-', label='Cont. states')
plt.plot(df_M28['step'], df_M28['ratio makespan'], linestyle='-', label='Discrete float M=28')
plt.plot(df_M21['step'], df_M21['ratio makespan'], linestyle='-', label='Discrete float M=21')
plt.plot(df_M14['step'], df_M14['ratio makespan'], linestyle='-', label='Discrete float M=14')
plt.plot(df_M7['step'], df_M7['ratio makespan'], linestyle='-', label='Discrete float M=7')



plt.xlabel('Training Steps')
plt.ylabel('Ratio Error Makespan')
plt.ylim([1, 1.9])
plt.title('Good tasks scaling inputs')
plt.legend()
plt.grid(True)
plt.show()