import pandas as pd

# 讀取CSV檔案
df = pd.read_csv("/work/u3359154/syn2real/SYN2REAL/cluster_results_folds/fold1_clusters.csv")

# 計算cluster欄的每個值出現的次數，並轉換為百分比
cluster_counts = df['cluster'].value_counts(normalize=True).sort_index()

# 確保0~7都被列出，即使某些值沒有出現
for i in range(8):
    percent = cluster_counts.get(i, 0) * 100
    print(f"Cluster {i}: {percent:.2f}%")