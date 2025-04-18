import datasets
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import hdbscan
import numpy as np
from tqdm import tqdm
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ========== 參數 ==========
CLUSTERING_ALGORITHM = "kmeans"  # 可選: kmeans / agglomerative / dbscan / hdbscan
CLUSTER_NUMBERS = [4, 8, 16, 32]
SCENARIOS = ["general", "weather", "qa", "social", "music", "datetime", "alarm", "email", "recommendation"]
MODEL_NAME = "all-mpnet-base-v2"
# Alibaba-NLP/gte-Qwen2-1.5B-instruct
# intfloat/multilingual-e5-large-instruct
# Lajavaness/bilingual-embedding-large
# NovaSearch/stella_en_1.5B_v5
BATCH_SIZE = 128

# ========== 資料載入與預處理 ==========
print("載入 SLURP 資料集 ...")
real_dataset = datasets.load_dataset("jacksukk/slurp")

train_dataset = real_dataset["train"].filter(lambda x: x["scenario"] in SCENARIOS)

# 去重
unique_texts = list(dict.fromkeys(train_dataset["text"]))
print(f"篩選後且去重的訓練資料數量：{len(unique_texts)}")

# ========== 嵌入模型 ==========
print("載入 SentenceTransformer 模型 ...")
model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

def encode_texts(texts):
    embeddings = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch = texts[i:i + BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# 嵌入訓練資料
train_embeddings = encode_texts(unique_texts)

# ========== 建立分群模型 ==========
clustering_models = {}
print(f"\n訓練 {CLUSTERING_ALGORITHM.upper()} 模型 ...")
for k in CLUSTER_NUMBERS:
    if CLUSTERING_ALGORITHM == "kmeans":
        model_instance = KMeans(n_clusters=k, random_state=42, n_init="auto")
        model_instance.fit(train_embeddings)
    elif CLUSTERING_ALGORITHM == "agglomerative":
        model_instance = AgglomerativeClustering(n_clusters=k)
    elif CLUSTERING_ALGORITHM == "dbscan":
        model_instance = DBSCAN(eps=1.0, min_samples=5)
    elif CLUSTERING_ALGORITHM == "hdbscan":
        model_instance = hdbscan.HDBSCAN(min_cluster_size=max(5, len(train_embeddings) // k))
    else:
        raise ValueError(f"未知的分群演算法：{CLUSTERING_ALGORITHM}")
    
    clustering_models[k] = model_instance
print("模型訓練完成。")

# ========== 嵌入與分群預測 ==========
def cluster_dataset(dataset, clustering_models):
    texts = dataset["text"]
    all_embeddings = encode_texts(texts)

    for k in CLUSTER_NUMBERS:
        model = clustering_models[k]
        if hasattr(model, "predict"):
            labels = model.predict(all_embeddings)
        else:
            labels = model.fit_predict(all_embeddings)
        dataset = dataset.add_column(f"cluster_{k}", labels)

    return dataset

# ========== 分群預測 ==========
train_dataset = cluster_dataset(real_dataset["train"], clustering_models)
val_dataset = cluster_dataset(real_dataset["devel"], clustering_models)
test_dataset = cluster_dataset(real_dataset["test"], clustering_models)

# ========== 評估 ==========
def evaluate_clustering(dataset, cluster_numbers, set_name, label_column="scenario"):
    print(f"\n{set_name} 分群品質評估：")
    for k in cluster_numbers:
        true_labels = dataset[label_column]
        pred_labels = dataset[f"cluster_{k}"]
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        print(f"[K={k}] ARI: {ari:.4f} | NMI: {nmi:.4f}")

evaluate_clustering(train_dataset, CLUSTER_NUMBERS, "train")
evaluate_clustering(val_dataset, CLUSTER_NUMBERS, "valid")
evaluate_clustering(test_dataset, CLUSTER_NUMBERS, "test")

exit(0)

# ======================================================
# 4. 批次預測分群標籤 (SLURP Synthetic Bark)
# ======================================================
from collections import Counter

print("製作切分用文字字典 ...")
split_lookup = {}
for split_name, dataset in {"train": real_dataset["train"], "devel": real_dataset["devel"], "test": real_dataset["test"]}.items():
    for text in dataset["text"]:
        split_lookup[text] = split_name

# 統計每個 split_name 出現的次數
split_counts = Counter(split_lookup.values())

# 計算總數
total = sum(split_counts.values())

# 計算比例
split_ratios = {key: value / total for key, value in split_counts.items()}

print("Split counts:", split_counts)
print("Split ratios:", split_ratios)
print("下載 slurp_synthetic_bark 資料集 ...")
synthetic_dataset = datasets.load_dataset("jacksukk/slurp_synthetic_bark")

# 用來存放每個 split 的結果
updated_synthetic_dataset = {}

for split_name, split_data in synthetic_dataset.items():
    print(f"\n處理 split: {split_name}")

    texts = split_data["text"]
    cluster_results = {f'cluster_{k}': [] for k in cluster_numbers}
    new_splits = []  # 存放 split 標籤

    for i in tqdm(range(0, len(texts), 128), desc=f"Encoding {split_name}"):
        batch = texts[i : i + 128]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)

        for k in cluster_numbers:
            cluster_labels = kmeans_models[k].predict(batch_embeddings)
            cluster_results[f'cluster_{k}'].extend(cluster_labels)

    # 依據 real_dataset["train"], ["devel"], ["test"] 決定 split
    for text in texts:
        new_splits.append(split_lookup.get(text, "unknown"))  # 如果找不到則標記為 "unknown"

    # 加入分群標籤與 split 欄位到 Dataset
    for k in cluster_numbers:
        split_data = split_data.add_column(f'cluster_{k}', cluster_results[f'cluster_{k}'])

    split_data = split_data.add_column("split", new_splits)  # 新增 split 欄位

    updated_synthetic_dataset[split_name] = split_data

print("已成功更新 synthetic_dataset，包含 cluster 標籤與 split 欄位！")

for split_name, split_data in updated_synthetic_dataset.items():
    print(f"\nSplit: {split_name}")
    
    split_counts = Counter(split_data["split"])
    total = sum(split_counts.values())
    
    for split_type in ["train", "devel", "test"]:
        percentage = (split_counts[split_type] / total) * 100 if split_type in split_counts else 0
        print(f"{split_type}: {split_counts[split_type]} ({percentage:.2f}%)")

# ======================================================
# 6. 轉換為 Hugging Face Dataset 格式並儲存 Parquet
# ======================================================
from huggingface_hub import HfApi, Repository
import os

# 准备 Hugging Face API 对象
hf_api = HfApi()

# 设置你的数据集存储库名称
dataset_name = "slurp_clustered_split_dataset"
repo_id = f"caster97/{dataset_name}"  # 这个就是你在 Hugging Face 上创建的 repo 名称

# 如果需要，首先创建一个新的 dataset repo（如果你还没有创建的话）
hf_api.create_repo(repo_id, repo_type="dataset")  # 需要删除注释行来创建 repo

# 创建本地目录来存储文件并将文件复制到该目录
local_dir = f"./{dataset_name}"

# 确保文件夹存在
os.makedirs(local_dir, exist_ok=True)

# 保存数据集到本地并上传到 Hugging Face
def save_and_upload_dataset(dataset, filename, repo_id):
    dataset.to_parquet(filename+"-00000-of-00001.parquet")  # 转换为 parquet 格式
    # 上传文件到 Hugging Face 的 /data 目录
    hf_api.upload_file(
        path_or_fileobj=f"{filename}-00000-of-00001.parquet",
        path_in_repo=f"data/{filename}-00000-of-00001.parquet",  # 指定文件上传到 /data 目录
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"上传文件 {filename} 成功!")

# 上传 synthetic 数据集
for split_name, dataset in updated_synthetic_dataset.items():
    save_and_upload_dataset(dataset, f"synth_{split_name}", repo_id)

# 上传不同的数据集 split
save_and_upload_dataset(train_dataset, "real_tr", repo_id)
save_and_upload_dataset(val_dataset, "real_v", repo_id)
save_and_upload_dataset(test_dataset, "real_te", repo_id)