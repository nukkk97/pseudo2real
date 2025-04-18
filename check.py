from datasets import load_dataset

# Load real dataset and count unique texts
real_dataset = load_dataset("jacksukk/slurp")
real_texts = set()

for split in ["train"]:  # Iterate over all available splits
    real_texts.update(real_dataset[split]["text"])  

print("Number of unique texts in real dataset (train split):", len(real_texts))

# Load synthetic dataset and count unique texts
synthetic_dataset = load_dataset("jacksukk/slurp_synthetic_bark")
synthetic_texts = set()

for split in synthetic_dataset.keys():  # Iterate over all available splits
    synthetic_texts.update(synthetic_dataset[split]["text"])  

print("Number of unique texts in synthetic dataset:", len(synthetic_texts))
