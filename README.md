# Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition

This repository provides the implementation of the **Pseudo2Real** parameter-space correction method for fine-tuning and evaluating Whisper ASR models on the multi-accent [AfriSpeech-200](https://huggingface.co/datasets/dlion168/afrispeech200_syn2real) benchmark.

## Contents

| File | Purpose |
|---|---|
| `train_afris_custom.py` | Fine-tunes a Whisper checkpoint on real transcripts, pseudo-labels, or a mix of both — optionally restricted to specific accents and/or pseudo-domain clusters. |
| `test_afris.py` | Builds a task-arithmetic "correction vector" from a source-domain model pair and applies it to a target-domain model, then evaluates WER. |

## Requirements

```bash
pip install -r requirements.txt
```

A CUDA GPU is expected. Both scripts pull from the Hugging Face dataset `dlion168/afrispeech200_syn2real`, which provides per-utterance audio, an `accent` label, a ground-truth `transcript`, and one or more Whisper-generated pseudo-label columns (e.g. `tiny_to`, `base_to`, `small_to`, `medium_to`, `large_to`).

Supported accents (`AFRIS_DOMAIN`): `yoruba`, `ijaw`, `afrikaans`, `idoma`, `setswana`, `igbo`, `swahili`, `hausa`, `zulu`, `twi`.

## Typical workflow

1. **Source domain, pseudo-labels** → `train_afris_custom.py --syn True ...` → `model_syn_anti`
2. **Source domain, real transcripts** → `train_afris_custom.py --syn False ...` → `model_anti`
3. **Target domain, pseudo-labels** → `train_afris_custom.py --syn True ...` → `model_target_syn`
4. **Correct + evaluate** → `test_afris.py` combines the three checkpoints (or a precomputed vector) and reports WER on the target domain's test set.

## 1. `train_afris_custom.py` — fine-tune Whisper

```bash
python train_afris_custom.py \
  --model_path openai/whisper-small \
  --synth_text small_to \
  --syn True \
  --domains "yoruba;igbo" \
  --configs ./configs/whisper_small.yaml
```

| Argument | Description |
|---|---|
| `--model_path` | Base Whisper checkpoint to fine-tune (HF Hub id or local path). |
| `--synth_text` | *(required)* Dataset column with the pseudo-label text to use as the target, e.g. `tiny_to` / `small_to` / `medium_to` / `large_to`. |
| `--syn` | `True` = train on pseudo-labels, `False` = train on the ground-truth `transcript`, `Mixed` = concatenate both. |
| `--domains` | Semicolon-separated accents to train on, e.g. `"yoruba;igbo"`. Omit to use all data. |
| `--cluster` | `1`, `4`, or `8` pseudo-domains to split the selected data into (the Pseudo2Real-SC / speaker-clustering setting). |
| `--current_pseudo` | Which pseudo-domain cluster (0-indexed) to train on this run. |
| `--fold` | Fold index, used when looking up the matching cluster-assignment CSV. |
| `--random` | `True` to use a random cluster assignment instead of the learned clusters (baseline). |
| `--filter_list` | Optional path to a text file of `audio_id`s to exclude from training. |
| `--t5_model_path` | Optional T5 checkpoint used to clean up noisy pseudo-label text before training on it. |
| `--configs` | YAML file of `Seq2SeqTrainingArguments` (LR, batch size, eval strategy, etc.). `max_steps` is hardcoded to 70000 and early-stopping patience to 20 regardless of the YAML. |

> **Note:** when `--domains` is set together with `--cluster > 1` (or `--random True`), the script looks up cluster assignments from a hardcoded path (`.../cluster_results_folds/fold{N}_clusters.csv` / `..._random.csv`) inside `train_afris_custom.py`. Point that path at your own cluster CSVs, or use `--cluster 1` to skip clustering entirely.

Checkpoints and the processor are saved to `./outputs/<run_name>/`, where `run_name` is auto-generated from your settings. Passing an existing `outputs/...` checkpoint as `--model_path` is treated as resuming training: the run name gets a `_continue` suffix and the learning rate is divided by 10.

## 2. `test_afris.py` — correction + evaluation

Computes `correction = model_anti − model_syn_anti` (real-label weights minus pseudo-label weights, in a source domain) and adds `weight × correction` onto a target-domain pseudo-label model, then reports WER on the AfriSpeech-200 test split.

**a) Evaluate one checkpoint directly (no merging):**
```bash
python test_afris.py \
  --model_path ./outputs/whisper_afris_synth_fold-0_cluster-0-of-1_small \
  --domain yoruba igbo \
  --output_name results.txt
```

**b) Merge on the fly from three checkpoints:**
```bash
python test_afris.py \
  --model_syn_anti ./outputs/source_pseudo_label_model \
  --model_anti ./outputs/source_real_label_model \
  --model_target_syn ./outputs/target_pseudo_label_model \
  --weight 1.0 \
  --domain hausa \
  --output_name results.txt
```

**c) Apply a precomputed correction vector:**
```bash
python test_afris.py \
  --model_target_syn ./outputs/target_pseudo_label_model \
  --model_vector ./vectors/source_correction.safetensors \
  --weight 0.5 \
  --domain zulu \
  --output_name results.txt
```

| Argument | Description |
|---|---|
| `--model_path` | Single checkpoint to evaluate as-is (vanilla mode, no merging). |
| `--model_syn_anti` | Source-domain checkpoint fine-tuned on pseudo-labels. |
| `--model_anti` | Source-domain checkpoint fine-tuned on ground-truth transcripts. |
| `--model_target_syn` | Target-domain checkpoint fine-tuned on pseudo-labels — the model being corrected. |
| `--model_vector` | Path to a precomputed correction vector (`.safetensors`); skips having to reload `model_syn_anti` / `model_anti`. |
| `--domain` | One or more accents to filter the test set to, e.g. `--domain yoruba igbo`. |
| `--weight` | Scale factor applied to the correction vector before adding it (typically 0.0–1.0). |
| `--output_name` | Text file that each run appends a `<domain> <weight> <WER>` line to. |

> **Note:** decoding uses `processor.get_decoder_prompt_ids(language="en", task="translate")`. `train_afris_custom.py` uses `task="transcribe"` instead — worth confirming the mismatch is intentional before comparing WER numbers across the two scripts.

## Paper

**Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition**
Yi-Cheng Lin, Yu-Hsuan Li Liang, Hsuan Su, Tzu-Quan Lin, Shang-Tse Chen, Yun-Nung Chen, Hung-yi Lee — [arXiv:2510.08047](https://arxiv.org/abs/2510.08047)

**Abstract (paraphrased):** Speech recognizers tend to struggle on accents or domains that are underrepresented in training data, and pseudo-labeling — fine-tuning on a model's own transcriptions of unlabeled audio — is a common workaround. The catch is that pseudo-labels tend to carry consistent, accent-linked mistakes that simple confidence filtering doesn't catch. The fix proposed here works entirely in parameter space and needs no labels in the eventual target domain: wherever a source domain happens to have both real transcripts and pseudo-labels available, two otherwise-identical models are trained, one on each label type, and their weights are simply subtracted to give a reusable "correction vector" that stands in for whatever bias the pseudo-labels introduce. Adding a scaled copy of that vector onto a model trained on pseudo-labels for a new domain nudges its behavior back toward what a real-label model would have produced. On the AfriSpeech-200 benchmark spanning ten African accents, this lifts Whisper models of several sizes, cutting relative word-error-rate by as much as 35% for Whisper-tiny. A follow-on variant, Pseudo2Real-SC, splits the source data by speaker cluster and computes one correction vector per subgroup for extra robustness — the mechanism mirrored by the `--cluster` / `--current_pseudo` options in `train_afris_custom.py`.

## Citation

If you use this code or the Pseudo2Real method, please cite:

```bibtex
@misc{lin2026pseudo2realtaskarithmeticpseudolabel,
      title={Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition}, 
      author={Yi-Cheng Lin and Yu-Hsuan Li Liang and Hsuan Su and Tzu-Quan Lin and Shang-Tse Chen and Yun-Nung Chen and Hung-yi Lee},
      year={2026},
      eprint={2510.08047},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2510.08047}, 
}
```