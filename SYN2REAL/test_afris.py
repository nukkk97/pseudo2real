from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
from evaluate import load
from argparse import ArgumentParser
from safetensors.torch import load_file as load_safetensors
import pandas as pd

def map_to_pred(batch):
    audio = batch["audio"]
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features
    batch["reference"] = processor.tokenizer._normalize(batch['transcript'])  # Use 'transcript' for the Afrispeech dataset

    with torch.no_grad():
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="translate")
        predicted_ids = model.generate(input_features.to("cuda"), forced_decoder_ids=forced_decoder_ids)[0]
    transcription = processor.decode(predicted_ids, skip_special_tokens=False)
    batch["prediction"] = processor.tokenizer._normalize(transcription)
    return batch

def merge(model_syn_anti, model_anti, model_target_syn, vector_dict, args):
    if vector_dict is None:
        for p1, p2 in zip(model_syn_anti.parameters(), model_anti.parameters()):
            p2.data -= p1.data
        for p1, p2 in zip(model_anti.parameters(), model_target_syn.parameters()):
            p2.data += args.weight * p1.data
    else:
        model_keys = model_target_syn.state_dict().keys()
        named_params = dict(model_target_syn.named_parameters())
        for key in model_keys:
            if key in vector_dict and key in named_params:
                named_params[key].data += args.weight * vector_dict[key]
    return model_target_syn

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None) # base model (not used when merging, only usable in vanilla mode)
    parser.add_argument('--model_syn_anti', type=str, default=None) # source domain synth model
    parser.add_argument('--model_anti', type=str, default=None) # source domain real model
    parser.add_argument('--model_target_syn', type=str, default=None) # target domain synth model
    parser.add_argument('--model_vector', type=str, default=None) # pre-calculated source domain syn2real vector
    parser.add_argument('--domain', nargs='+', type=str, default=None) # target domain
    parser.add_argument('--weight', type=float, default=1.0) # merging weight 0.0-1.0
    parser.add_argument('--output_name', type=str, default='result.txt') # output name
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset("dlion168/afrispeech200_syn2real", split="test")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    if args.domain is not None:
        dataset = dataset.filter(lambda example: example["accent"] in args.domain)

    # Model loading logic
    if all(x is None for x in [args.model_syn_anti, args.model_anti, args.model_target_syn, args.model_vector]):
        print("Loading model without merge")
        model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to("cuda")
        processor = WhisperProcessor.from_pretrained(args.model_path)
    elif args.model_vector is None:
        print("Loading models and merging without vector")
        model_syn_anti = WhisperForConditionalGeneration.from_pretrained(args.model_syn_anti)
        processor = WhisperProcessor.from_pretrained(args.model_syn_anti)
        model_anti = WhisperForConditionalGeneration.from_pretrained(args.model_anti)
        model_target_syn = WhisperForConditionalGeneration.from_pretrained(args.model_target_syn)
        model = merge(model_syn_anti, model_anti, model_target_syn, None, args).to("cuda")
        args.model_path = 'merge'
    else:
        print("Loading model and applying vector")
        model_target_syn = WhisperForConditionalGeneration.from_pretrained(args.model_target_syn)
        processor = WhisperProcessor.from_pretrained(args.model_target_syn)
        named_params = dict(model_target_syn.named_parameters())
        vector_dict = load_safetensors(args.model_vector)
        for key in named_params:
            if key in vector_dict:
                if named_params[key].data.shape != vector_dict[key].shape:
                    raise ValueError(f"Shape mismatch for {key}: {named_params[key].shape} vs {vector_dict[key].shape}")
                named_params[key].data += args.weight * vector_dict[key]
            else:
                print(f"Warning: Key '{key}' not found in vector — skipping.")
        
        model = merge(None, None, model_target_syn, vector_dict, args).to("cuda")
        args.model_path = 'merge'

    model.config.forced_decoder_ids = None
    result = dataset.map(map_to_pred)
    wer = load("wer")
    with open(f"{args.output_name}", "a", encoding="utf-8") as f:
        f.write(f"{args.domain} {args.weight} {100 * wer.compute(references=result['transcript'], predictions=result['prediction']):.2f}\n")