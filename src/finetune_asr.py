# scripts for fine-tuning
"""
finetune_asr.py
---------------
Minimal training recipe to fine-tune Wav2Vec2 (CTC) on a domain dataset using Hugging Face.

This script is a template — adapt dataset loading and hyperparameters for your compute.
It uses the `datasets` library and the `transformers` Trainer API.

Important:
 - Requires substantial GPU for reasonable speed.
 - Input dataset must include columns: "audio" (path or dict with 'array'/'sampling_rate') and "text".

Usage (example):
  python src/finetune_asr.py --train_jsonl data/train.jsonl --eval_jsonl data/valid.jsonl --output_dir models/wav2vec2-domain
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import evaluate
from datasets import load_dataset, Audio, Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    DataCollatorWithPadding,
)

# Utility: compute WER
wer_metric = evaluate.load("wer")


@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: bool = True

    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        # labels
        labels = [f["labels"] for f in features]
        # pad labels
        batch_labels = self.processor.tokenizer.pad(
            {"input_ids": labels}, return_tensors="pt"
        )["input_ids"]
        # replace padding label id's by -100 to ignore in loss
        batch_labels[batch_labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = batch_labels
        return batch


def prepare_dataset(dataset, processor, target_sample_rate: int = 16000):
    """Resample audio, extract input_values, tokenize labels"""
    def prepare_example(example):
        # load audio with datasets Audio feature if not already loaded
        audio = example["audio"]
        if isinstance(audio, dict) and "array" in audio:
            speech_array = np.asarray(audio["array"])
            sampling_rate = audio["sampling_rate"]
        else:
            # fallback: load file manually via librosa (not added here)
            raise ValueError("Audio not in dataset in expected format (dict with 'array').")

        # resample if needed (datasets Audio auto does this if set)
        if sampling_rate != target_sample_rate:
            import librosa
            speech_array = librosa.resample(speech_array, orig_sr=sampling_rate, target_sr=target_sample_rate)
            sampling_rate = target_sample_rate

        input_values = processor.feature_extractor(speech_array, sampling_rate=target_sample_rate).input_values[0]
        with processor.as_target_processor():
            labels = processor.tokenizer(example["text"]).input_ids
        return {"input_values": input_values, "labels": labels}

    return dataset.map(prepare_example, remove_columns=dataset.column_names, num_proc=1)


def main(args):
    # Load datasets (expect jsonl with audio path field named "audio_filepath" and "text")
    data_files = {}
    if args.train_jsonl:
        data_files["train"] = args.train_jsonl
    if args.eval_jsonl:
        data_files["validation"] = args.eval_jsonl

    ds = load_dataset("json", data_files=data_files)
    # Cast 'audio' column
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000))

    # Load pretrained processor (feature_extractor + tokenizer)
    processor = Wav2Vec2Processor.from_pretrained(args.pretrained_model)
    model = Wav2Vec2ForCTC.from_pretrained(args.pretrained_model)

    # Prepare dataset
    prepared = {}
    for split in ds:
        prepared[split] = ds[split].map(lambda ex: {"audio": ex["audio"], "text": ex["text"]})

    # Prepare examples: this uses the Audio feature results and the processor
    train_ds = prepared.get("train")
    val_ds = prepared.get("validation")

    train_prepared = prepare_dataset(train_ds, processor) if train_ds else None
    val_prepared = prepare_dataset(val_ds, processor) if val_ds else None

    data_collator = DataCollatorCTCWithPadding(processor=processor)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy="steps" if args.eval_steps else "no",
        eval_steps=args.eval_steps or 500,
        save_steps=args.save_steps or 500,
        logging_steps=100,
        gradient_accumulation_steps=args.grad_accum or 1,
        fp16=not args.no_fp16,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        save_total_limit=3,
    )

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids
        # replace -100 in labels
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_prepared,
        eval_dataset=val_prepared,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,  # not strictly required
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune wav2vec2 on domain data")
    parser.add_argument("--train_jsonl", type=str, default=None, help="train jsonl path with audio_filepath and text")
    parser.add_argument("--eval_jsonl", type=str, default=None, help="eval jsonl path")
    parser.add_argument("--pretrained_model", type=str, default="facebook/wav2vec2-base-960h")
    parser.add_argument("--output_dir", type=str, default="models/wav2vec2-domain")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--no_fp16", action="store_true")
    args = parser.parse_args()
    main(args)
