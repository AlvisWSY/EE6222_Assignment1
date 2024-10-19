import pandas as pd

# Define the file path
file_path = '/media/user/volume2/students/s124md209_01/WangShengyuan/6222/Assignment1/result/results.csv'
save_path = '/media/user/volume2/students/s124md209_01/WangShengyuan/6222/Assignment1/result/results_m.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Replace empty values with 'N/A'
df.fillna('N/A', inplace=True)

# Replace 'NA' with 'TimeOutError'
df.replace('NA', 'TimeOutError', inplace=True)
# Filter rows where the first column is 'cifar10' and the last column is 'N/A'
filtered_df = df[(df.iloc[:, 0] == 'cifar10') & (df.iloc[:, -1] == 'N/A')]

# Update the last column from 'N/A' to 'TimeOutError' for the filtered rows
df.loc[filtered_df.index, df.columns[-1]] = 'TimeOutError'
# Write the modified DataFrame back to the CSV file
df.to_csv(save_path, index=False)