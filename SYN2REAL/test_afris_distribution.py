from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
from argparse import ArgumentParser
from safetensors.torch import load_file as load_safetensors
import pandas as pd
import Levenshtein
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import seaborn as sns
import os

###############################################################
# WORD-LEVEL EDIT DISTANCE STAT COUNTERS
###############################################################
subs = Counter()  # (ref_word, pred_word)
ins = Counter()   # pred_word
dels = Counter()  # ref_word

###############################################################
# PER-UTTERANCE STORAGE
###############################################################
per_utt_rows = []

###############################################################
# HELPER FUNCTIONS
###############################################################
def analyze_ops(ref, pred):
    ref_words = ref.split()
    pred_words = pred.split()
    ops = Levenshtein.editops(ref_words, pred_words)
    S = I = D = 0
    for op, i, j in ops:
        if op == "replace":
            subs[(ref_words[i], pred_words[j])] += 1
            S += 1
        elif op == "insert":
            ins[pred_words[j]] += 1
            I += 1
        elif op == "delete":
            dels[ref_words[i]] += 1
            D += 1
    return S, I, D

def compute_wer(ref, hyp):
    wer_metric = load("wer")
    return wer_metric.compute(references=[ref], predictions=[hyp])

def save_word_level_stats(subs, ins, dels, domain, weight):
    df_sub = pd.DataFrame([(a, b, c) for (a, b), c in subs.items()],
                          columns=["from","to","count"])
    df_sub["domain"] = str(domain)
    df_sub["weight"] = weight
    if os.path.exists("substitution_stats.csv"):
        df_sub.to_csv("substitution_stats.csv", mode="a", header=False, index=False)
    else:
        df_sub.to_csv("substitution_stats.csv", index=False)

    df_ins = pd.DataFrame([(w, c) for w, c in ins.items()],
                          columns=["inserted_word","count"])
    df_ins["domain"] = str(domain)
    df_ins["weight"] = weight
    if os.path.exists("insertion_stats.csv"):
        df_ins.to_csv("insertion_stats.csv", mode="a", header=False, index=False)
    else:
        df_ins.to_csv("insertion_stats.csv", index=False)

    df_del = pd.DataFrame([(w, c) for w, c in dels.items()],
                          columns=["deleted_word","count"])
    df_del["domain"] = str(domain)
    df_del["weight"] = weight
    if os.path.exists("deletion_stats.csv"):
        df_del.to_csv("deletion_stats.csv", mode="a", header=False, index=False)
    else:
        df_del.to_csv("deletion_stats.csv", index=False)


###############################################################
# MAP FUNCTION
###############################################################
def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt"
    ).input_features

    reference = processor.tokenizer._normalize(batch["transcript"])
    batch["reference"] = reference

    if "text_whisper-large-v2" in batch:
        pseudo_label = processor.tokenizer._normalize(batch["text_whisper-large-v2"])
    else:
        pseudo_label = reference
    batch["pseudo_label"] = pseudo_label

    with torch.no_grad():
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language="en", task="translate"
        )
        predicted_ids = model.generate(
            input_features.to("cuda"),
            forced_decoder_ids=forced_decoder_ids
        )[0]

    prediction = processor.decode(predicted_ids, skip_special_tokens=False)
    prediction = processor.tokenizer._normalize(prediction)
    batch["prediction"] = prediction

    S, I, D = analyze_ops(reference, prediction)

    model_wer = compute_wer(reference, prediction)
    label_error = compute_wer(reference, pseudo_label)

    per_utt_rows.append({
        "id": batch["id"] if "id" in batch else batch["audio"]["path"],
        "accent": batch.get("accent", "NA"),
        "reference": reference,
        "prediction": prediction,
        "pseudo_label": pseudo_label,
        "model_wer": model_wer,
        "label_error": label_error,
        "S": S,
        "I": I,
        "D": D,
        "ref_len": len(reference.split())
    })

    return batch

###############################################################
# MERGE VECTORS
###############################################################
def merge(model_syn_anti, model_anti, model_target_syn, vector_dict, args):
    if vector_dict is None:
        for p1, p2 in zip(model_syn_anti.parameters(), model_anti.parameters()):
            p2.data -= p1.data
        for p1, p2 in zip(model_anti.parameters(), model_target_syn.parameters()):
            p2.data += args.weight * p1.data
    else:
        named_params = dict(model_target_syn.named_parameters())
        for key in model_target_syn.state_dict().keys():
            if key in vector_dict and key in named_params:
                named_params[key].data += args.weight * vector_dict[key]
    return model_target_syn

###############################################################
# MAIN
###############################################################
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_syn_anti', type=str, default=None)
    parser.add_argument('--model_anti', type=str, default=None)
    parser.add_argument('--model_target_syn', type=str, default=None)
    parser.add_argument('--model_vector', type=str, default=None)
    parser.add_argument('--domain', nargs='+', type=str, default=None)
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--output_name', type=str, default='result.txt')
    args = parser.parse_args()

    dataset = load_dataset("dlion168/afrispeech200_syn2real", split="test")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    if args.domain is not None:
        dataset = dataset.filter(lambda example: example["accent"] in args.domain)

    if all(x is None for x in [args.model_syn_anti, args.model_anti, args.model_target_syn, args.model_vector]):
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to("cuda")
        processor = WhisperProcessor.from_pretrained(args.model_path)

    elif args.model_vector is None:
        model_syn_anti = WhisperForConditionalGeneration.from_pretrained(args.model_syn_anti)
        processor = WhisperProcessor.from_pretrained(args.model_syn_anti)
        model_anti = WhisperForConditionalGeneration.from_pretrained(args.model_anti)
        model_target_syn = WhisperForConditionalGeneration.from_pretrained(args.model_target_syn)
        model = merge(model_syn_anti, model_anti, model_target_syn, None, args).to("cuda")
        args.model_path = "merge"

    else:
        model_target_syn = WhisperForConditionalGeneration.from_pretrained(args.model_target_syn)
        processor = WhisperProcessor.from_pretrained(args.model_target_syn)
        vector_dict = load_safetensors(args.model_vector)

        named_params = dict(model_target_syn.named_parameters())
        for key in named_params:
            if key in vector_dict:
                named_params[key].data += args.weight * vector_dict[key]

        model = merge(None, None, model_target_syn, vector_dict, args).to("cuda")
        args.model_path = "merge"

    model.config.forced_decoder_ids = None
    result = dataset.map(map_to_pred)

    wer = load("wer")
    model_wer_val = 100 * wer.compute(references=result['reference'], predictions=result['prediction'])
    with open(args.output_name, "a", encoding="utf-8") as f:
        f.write(f"{args.domain} {args.weight} {model_wer_val:.2f}\n")

    save_word_level_stats(subs, ins, dels, args.domain, args.weight)

    df_per_utt = pd.DataFrame(per_utt_rows)
    df_per_utt["domain"] = str(args.domain)
    df_per_utt["weight"] = args.weight
    if os.path.exists("per_utterance_errors.csv"):
        df_per_utt.to_csv("per_utterance_errors.csv", mode="a", header=False, index=False)
    else:
        df_per_utt.to_csv("per_utterance_errors.csv", index=False)

    ###############################################################
    # SID TOTAL AND OUTPUT sid_summary.csv
    ###############################################################
    total_S = sum(row["S"] for row in per_utt_rows)
    total_I = sum(row["I"] for row in per_utt_rows)
    total_D = sum(row["D"] for row in per_utt_rows)

    sid_df = pd.DataFrame([{
        "domain": str(args.domain),
        "weight": args.weight,
        "total_sub": total_S,
        "total_ins": total_I,
        "total_del": total_D,
        "mean_ref_len": np.mean([r["ref_len"] for r in per_utt_rows]),
        "samples": len(per_utt_rows)
    }])

    if os.path.exists("sid_summary.csv"):
        sid_df.to_csv("sid_summary.csv", mode="a", header=False, index=False)
    else:
        sid_df.to_csv("sid_summary.csv", index=False)

    ###############################################################
    # HEATMAP sorted by count (TOP-K)
    ###############################################################
    TOP_K = 50
    top_pairs = subs.most_common(TOP_K)

    if len(top_pairs) > 0:
        ref_words = sorted(list({a for (a, b), _ in top_pairs}))
        pred_words = sorted(list({b for (a, b), _ in top_pairs}))

        heatmap_data = pd.DataFrame(0, index=ref_words, columns=pred_words)
        for (a, b), c in top_pairs:
            heatmap_data.loc[a, b] = c

        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, cmap="viridis", annot=False)
        plt.xlabel("Prediction Word")
        plt.ylabel("Reference Word")
        plt.title(f"Top-{TOP_K} Substitution Heatmap [{args.domain}, weight={args.weight}]")
        plt.tight_layout()
        plt.savefig("word_heatmap_sorted.png", dpi=300)

    ###############################################################
    # SCATTER PLOT: label_error vs model_wer
    ###############################################################
    x = df_per_utt["label_error"].values.reshape(-1,1)
    y = df_per_utt["model_wer"].values
    lr = LinearRegression().fit(x, y)
    slope = lr.coef_[0]; r2 = lr.score(x, y)

    plt.figure(figsize=(6,5))
    plt.scatter(x, y, alpha=0.3, s=10)
    xs = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
    plt.plot(xs, lr.predict(xs), color='red', label=f"slope={slope:.3f}, R²={r2:.3f}")
    plt.xlabel("Label error (pseudo vs ground truth WER)")
    plt.ylabel("Model WER")
    plt.legend()
    plt.tight_layout()
    plt.savefig("label_error_vs_model_wer.png", dpi=200)

    print(f"Model WER: {model_wer_val:.2f}")
    print("SID summary, per-utterance CSV, word heatmap, and scatter plot saved.")