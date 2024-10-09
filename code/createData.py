import pandas as pd
from sklearn.datasets import load_wine

# 加载 Wine 数据集
wine = load_wine()

# 将特征数据和标签合并为一个 DataFrame
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine['label'] = wine.target

# 将 DataFrame 保存为 CSV 文件，路径为 ./data/wine_data.csv
csv_path = '../data/wine_data.csv'
df_wine.to_csv(csv_path, index=False)

print(f"Wine 数据集已保存为 CSV 文件：{csv_path}")
