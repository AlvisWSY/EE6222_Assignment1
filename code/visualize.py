import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the results data from CSV
file_path = '../result/results.csv'
df = pd.read_csv(file_path)

# Replace 'N/A' and missing values in 'Accuracy' column with 0 to indicate failure cases
df['Accuracy'] = df['Accuracy'].replace(['N/A', 'NaN'], 0).fillna(0).astype(float)

# Replace 'N/A' in 'Number of Components' with a placeholder value to avoid filtering out 'none' reduction method
df['Number of Components'] = df['Number of Components'].replace('N/A', -1).fillna(-1)

# Define function to plot grouped comparison using parent and subplots
def plot_grouped_comparison(df):
    sns.set(style="whitegrid")

    # Group by dataset, and create parent plots for each dataset
    datasets = df['Dataset'].unique()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(18, 6 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    fig.suptitle('Comparison of Reduction Methods and Classifiers Across Datasets', fontsize=20)

    # Iterate through each dataset
    for i, dataset in enumerate(datasets):
        dataset_df = df[df['Dataset'] == dataset]

        # Create a plot for the dataset
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            data=dataset_df,
            x='Reduction Method',
            y='Accuracy',
            hue='Classifier',
            palette='viridis',
            ci=None  # Remove error bars
        )
        plt.title(f'Performance Comparison for {dataset}', fontsize=16)
        plt.xlabel('Reduction Method')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.legend(title='Classifier', bbox_to_anchor=(1, 1), loc='upper left')
        
        # Set y-axis limits based on dataset
        if dataset == 'cifar10':
            plt.ylim(0.2, 0.5)
        else:
            plt.ylim(0.8, 1.0)
        
        # Remove vertical grid lines
        ax.grid(axis='x', linestyle='')
        
        plt.tight_layout()
        plt.savefig(f'../result/{dataset}_comparison_plot.png')
        plt.show()

# Plot the grouped comparison
plot_grouped_comparison(df)