from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import WhisperForConditionalGeneration, WhisperProcessor, EarlyStoppingCallback
import torch
from evaluate import load
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, HfArgumentParser
from datasets import Audio
import random
import pandas as pd
from transformers import T5ForConditionalGeneration, T5Tokenizer

random.seed(42)
metric = evaluate.load("wer")
MAX_INPUT_FRAMES = 1024

# processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="en", task='transcribe')
AFRIS_DOMAIN = {
'yoruba',
'ijaw',
'afrikaans',
'idoma',
'setswana',
'igbo',
'swahili',
'hausa',
'zulu',
'twi'
}
SUBSET_SIZE = {
    'train': 100,
    'dev': 20,
    'test': 20
}
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        labels = labels[:, :448]

        batch["labels"] = labels
        

        return batch
        
def correct_with_t5(text: str, model, tokenizer):
    """Use a T5 model to correct noisy synthetic text before training."""
    if model is None:
        return text  # No correction if model not provided

    # Tokenize input with prefix (optional)
    input_text = f"correct: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128)
    
    corrected = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected

def prepare_dataset(batch, processor, text_column, t5_model=None, t5_tokenizer=None):
    audio = batch["audio"]

    batch["input_features"] = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]

    text = batch[text_column]

    # Apply T5 correction if available
    if t5_model is not None and t5_tokenizer is not None:
        text = correct_with_t5(text, t5_model, t5_tokenizer)

    batch["text"] = processor.tokenizer._normalize(text)
    labels = processor.tokenizer(batch["text"]).input_ids

    if len(labels) > 1024:
        return None
    batch["labels"] = labels
    return batch

# def prepare_dataset(batch, processor):
#     audio = batch["audio"]
#     batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
#     batch["labels"] = processor.tokenizer(batch["text"]).input_ids
#     print(processor.tokenizer(batch["text"]).input_ids)
#     print(processor.tokenizer.decode(batch["labels"], skip_special_tokens=False))
#     return batch

def is_short_enough(example):
    # 先抽特徵再判斷長度
    audio = example["audio"]
    input_features = processor.feature_extractor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    return len(input_features) <= MAX_INPUT_FRAMES

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str = [processor.tokenizer._normalize(s) for s in pred_str]
    label_str = [processor.tokenizer._normalize(s) for s in label_str]
    print('pred: ', pred_str)
    print('label: ', label_str)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}
if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--domains', type=str, default=None) # source domain
    parser.add_argument('--syn', type=str, default="True") # True: use ASR text, False: use original text, Mixed: use both
    parser.add_argument('--fold', type=int, default=0) # fold 0: select provided by --domains / fold 1: otherwise
    parser.add_argument('--cluster', type=int, choices=[1, 4, 8], default=8) # number of pseudo domains
    parser.add_argument('--current_pseudo', type=int, default=0) # current train on pseudo domain index
    parser.add_argument('--model_path', type=str, default="openai/whisper-small")
    parser.add_argument('--synth_text', type=str, required=True)
    parser.add_argument('--configs', type=str, default="/work/u3359154/syn2real/SYN2REAL/configs/whisper_small.yaml")
    parser.add_argument('--random', type=str, default="False") # whether to use random cluster
    parser.add_argument('--filter_list', type=str, default=None) # filter audio_ids
    parser.add_argument('--t5_model_path', type=str, default=None,
                    help="Optional: Path to pretrained T5 model for text correction before Whisper training.")
    args = parser.parse_args()
    
    print('loading model')
    args.domains = args.domains.split(';')
    args.domains = [d.strip() for d in args.domains if d.strip() != '']

    model = WhisperForConditionalGeneration.from_pretrained(args.model_path, device_map="auto")
    # model.config.dropout = 0.1
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    print('loading data')
    print('train_domains:', args.domains)
    processor = WhisperProcessor.from_pretrained(args.model_path, task='transcribe')
    # dataset = load_dataset("marcel-gohsen/slurp", use_auth_token=False, cache_dir="/work/b04203058/huggingface_hub")
    # dataset = load_dataset("audiofolder", data_files=data_files)
    dataset = load_dataset("dlion168/afrispeech200_syn2real")
    run_name = f'whisper_afris_{"synth" if args.syn == "True" else "mixed" if args.syn == "Mixed" else "real"}'
    training_args = HfArgumentParser(Seq2SeqTrainingArguments).parse_yaml_file(args.configs)[0]

    if args.t5_model_path is not None:
        print(f"Loading T5 correction model from {args.t5_model_path}")
        t5_tokenizer = T5Tokenizer.from_pretrained(args.t5_model_path)
        t5_model = T5ForConditionalGeneration.from_pretrained(args.t5_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        t5_model.eval()
        run_name += "_t5corrected"
    else:
        t5_model = None
        t5_tokenizer = None

    # 2. Optional: read audio_id filter list if provided
    clean_dataset = {}

    if args.filter_list is not None and len(args.filter_list.strip()) > 0:
        print(f"Applying filter list from: {args.filter_list}")
        with open(args.filter_list, "r") as f:
            filter_list = set(line.strip() for line in f if line.strip())

        # 3. 過濾每個 split
        for split in dataset.keys():
            clean_dataset[split] = dataset[split].filter(
                lambda x: x["audio_id"] not in filter_list
            )
    else:
        print("No filter list provided — using full dataset.")
        clean_dataset = dataset

    clean_dataset = DatasetDict(clean_dataset)


    if args.domains:
        selected = args.domains
        domain_dict = {d: 1 for d in selected}
        # 若 cluster > 1 才需要載入 cluster.csv 檔
        # 預設空 mapping
        id_to_cluster = {}

        # 如果需要 cluster 過濾才讀取檔案
        if args.random == "True":
            run_name += "_random"
            cluster_file = f"/work/u3359154/syn2real/SYN2REAL/cluster_results_folds/fold{args.fold}_random.csv"
            cluster_df = pd.read_csv(cluster_file)
            # 保證 id 是字串避免 key 錯失配
            id_to_cluster = {str(i): c for i, c in zip(cluster_df['audio_id'], cluster_df['cluster'])}
        elif args.cluster > 1:
            cluster_file = f"/work/u3359154/syn2real/SYN2REAL/cluster_results_folds/fold{args.fold}_clusters.csv"
            cluster_df = pd.read_csv(cluster_file)
            # 保證 id 是字串避免 key 錯失配
            id_to_cluster = {str(i): c for i, c in zip(cluster_df['audio_id'], cluster_df['cluster'])}

        def filter_func(example):
            # accent 不在 domain_dict 直接過濾
            if example["accent"] not in domain_dict:
                return False

            # cluster=1：只要 accent 符合就保留
            if args.cluster == 1:
                return True

            # cluster>1：必須匹配到相同 cluster
            audio_id = str(example["audio_id"])
            cluster = id_to_cluster.get(audio_id, None)
            return cluster == args.current_pseudo

        # 無論 cluster=1 或 >1 都能用同一個 filter_func
        for split in ['train', 'dev']:
            clean_dataset[split] = clean_dataset[split].filter(
                filter_func,
                load_from_cache_file=False,
                num_proc=8
            )


    run_name = run_name + "_" + f"fold-{args.fold}" + "_" + f"cluster-{args.current_pseudo}-of-{args.cluster}"
        
    patience = 20
    training_args.max_steps = 70000
    # run_name += "_speech_t5"
    # run_name += "_subset"
    if "small" in args.synth_text:
        run_name += "_small_to"
    elif "medium" in args.synth_text:
        run_name += "_medium_to"
    elif "large" in args.synth_text:
        run_name += "_large_to"
    elif "tiny" in args.synth_text:
        run_name += "_tiny_to"
    elif "base" in args.synth_text:
        run_name += "_base_to"

    if "small" in args.model_path:
        run_name += "_small"
    elif "medium" in args.model_path:
        run_name += "_medium"
    elif "large" in args.model_path:
        run_name += "_large"
    elif "tiny" in args.model_path:
        run_name += "_tiny"
    elif "base" in args.model_path:
        run_name += "_base"
    
    if "outputs/" in args.model_path:
        run_name = args.model_path.split('/')[-1] + "_continue"
        training_args.learning_rate = training_args.learning_rate * 0.1

    if len(args.domains) == 1:
        run_name += "_" + args.domains[0]

    clean_dataset = clean_dataset.cast_column("audio", Audio(sampling_rate=16000))
    print(clean_dataset['train'][0])

    # shuffle
    clean_dataset = clean_dataset.shuffle(seed=42)

    if args.syn == "True":
        for split in ['train', 'dev']:
            def filter_long_labels(example):
                return len(processor.tokenizer(example[args.synth_text]).input_ids) <= 447
            orig_len = len(clean_dataset[split])
            clean_dataset[split] = clean_dataset[split].filter(is_short_enough)
            new_len = len(clean_dataset[split])
            print(f"{split}: removed {orig_len - new_len} / {orig_len} samples (>1024 frames)")
            if split == 'train':
                clean_dataset[split] = clean_dataset[split].filter(filter_long_labels).map(
                    prepare_dataset,
                    fn_kwargs={"processor": processor, "text_column": args.synth_text, "t5_model": t5_model, "t5_tokenizer": t5_tokenizer},
                    num_proc=1,
                    load_from_cache_file=False
                )
            else:
                clean_dataset[split] = clean_dataset[split].filter(filter_long_labels).map(
                    prepare_dataset,
                    fn_kwargs={"processor": processor, "text_column": args.synth_text, "t5_model": t5_model, "t5_tokenizer": t5_tokenizer},
                    num_proc=1,
                    load_from_cache_file=False
                )

    elif args.syn == "False":
        for split in ['train', 'dev']:
            def filter_long_labels(example):
                return len(processor.tokenizer(example["transcript"]).input_ids) <= 447
            orig_len = len(clean_dataset[split])
            clean_dataset[split] = clean_dataset[split].filter(is_short_enough)
            new_len = len(clean_dataset[split])
            print(f"{split}: removed {orig_len - new_len} / {orig_len} samples (>1024 frames)")
            clean_dataset[split] = clean_dataset[split].filter(filter_long_labels).map(
                prepare_dataset,
                fn_kwargs={"processor": processor, "text_column": args.synth_text, "t5_model": t5_model, "t5_tokenizer": t5_tokenizer},
                num_proc=1,
                load_from_cache_file=False
            )

    elif args.syn == "Mixed":
        for split in ['train', 'dev']:
            def filter_long_labels1(example):
                return len(processor.tokenizer(example["transcript"]).input_ids) <= 447
            def filter_long_labels2(example):
                return len(processor.tokenizer(example[args.synth_text]).input_ids) <= 447
            orig_len = len(clean_dataset[split])
            clean_dataset[split] = clean_dataset[split].filter(is_short_enough)
            new_len = len(clean_dataset[split])
            print(f"{split}: removed {orig_len - new_len} / {orig_len} samples (>1024 frames)")
            if split == 'train':
                ds_transcript = clean_dataset[split].filter(filter_long_labels1).map(
                    prepare_dataset,
                    fn_kwargs={"processor": processor, "text_column": args.synth_text, "t5_model": t5_model, "t5_tokenizer": t5_tokenizer},
                    num_proc=1,
                    load_from_cache_file=False
                )
                ds_whisper = clean_dataset[split].filter(filter_long_labels2).map(
                    prepare_dataset,
                    fn_kwargs={"processor": processor, "text_column": args.synth_text, "t5_model": t5_model, "t5_tokenizer": t5_tokenizer},
                    num_proc=1,
                    load_from_cache_file=False
                )
                clean_dataset[split] = concatenate_datasets([ds_transcript, ds_whisper])
            else:
                clean_dataset[split] = clean_dataset[split].filter(filter_long_labels1).map(
                    prepare_dataset,
                    fn_kwargs={"processor": processor, "text_column": args.synth_text, "t5_model": t5_model, "t5_tokenizer": t5_tokenizer},
                    num_proc=1,
                    load_from_cache_file=False
                )

    print("Number of datapoints in train set:", len(clean_dataset['train']))
    print("Number of datapoints in dev set:", len(clean_dataset['dev']))
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    training_args.output_dir=f"./outputs/{run_name}"
    # training_args.learning_rate = learning_rate
    training_args.run_name = run_name
    # training_args.save_safetensors=False

    model.gradient_checkpointing_disable()
    model.config.use_cache = False
    trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=clean_dataset["train"],
    eval_dataset=clean_dataset["dev"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
    callbacks = [EarlyStoppingCallback(early_stopping_patience=patience)],
    )

    processor.save_pretrained(training_args.output_dir)
    trainer.train()
    trainer.save_model(training_args.output_dir)
