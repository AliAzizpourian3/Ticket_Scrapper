import pandas as pd
import numpy as np
import optuna
import os
import sklearn as sk
from collections import Counter
from tqdm import tqdm
from collections import Counter, defaultdict

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)

# LLM Training
def format_sample(specific_events, date, threshold_gap):

    specific_events = specific_events[~specific_events["name"].isna()]
    specific_events = specific_events[~( (specific_events["name"] == "VehicleId") & (specific_events["name"] == "VendorName") )]
    subset_events = specific_events[(specific_events["event_time"] >= date - threshold_gap) & (specific_events["event_time"] <= date)]

    subset_events["name"] = subset_events["name"].str.replace(r'\d+','X',regex=True)
    subset_events["event_time"] = subset_events["event_time"] - date
    subset_events["event_time"] = subset_events["event_time"]
    subset_events["full_name"] = subset_events.apply(
          lambda row: (
              f"{row['name']} (severity: {row['severity']}) (type: {row['type']}) " +
              (f"(value: {row['value']}) " if pd.notna(row['value']) else "")  + (f"(source: {row['source']})")
          ),
          axis=1
      )

    return subset_events[["event_time", "full_name"]].sort_values("event_time").itertuples(index=False)

def rel_time(event_time, date):
    diff = date - event_time
    total_seconds = diff.total_seconds()
    total_hours = int(total_seconds // 3600)
    total_minutes = int(total_seconds // 60)
    minutes = total_minutes % 60
    d, h = divmod(total_hours, 24)
    if d > 0:
        return f"-{d}d {h}h{minutes}m"
    if total_hours > 0:
        return f"-{h}h{minutes}m"
    return f"-{minutes}m"

def compact_format_sample(specific_events, date, threshold_gap, JSON=True):

    subset_events = specific_events[(specific_events["event_time"] >= date - threshold_gap) & (specific_events["event_time"] <= date)].copy()

    subset_events["_day"] = (( date - subset_events['event_time']).dt.total_seconds() // 86400).astype(int)
    
    # lines = ([
    #     f"-{day} | {group.groupby("severity").size().to_dict()}"
    #     for day, group in subset_events.groupby("_day")
    # ])

    subset_events = subset_events[~( (subset_events["name"] == "VehicleId") | (subset_events["name"] == "VendorName")  )]


    if subset_events.empty:
        if JSON:
            return json.dumps({"events": []}, separators=(",", ":"))
        else:
            return "\nSEV      | TYPE   | SOURCE                              | NAME                                | VALUE    |  Daily Counts\n(none)"
    
    
    grouped = subset_events.groupby(["name", "severity", "source", "value", "type", "error_code"], observed=False, dropna=False).agg({"_day": list}).reset_index()
    lines = []

    if JSON:
        for _, row in grouped.iterrows():

            arr = "[" + ", ".join([f"{Counter(row['_day']).get(i, 0)}" for i in range(5, -1, -1)]) + "]"
            
            event_obj = {
                "s": str(row['severity'])[:4],  
                # "t": str(row['type'])[:3],       
                "c": str(row['source'])[:18],  
                "n": str(row['name']),      
                "d": arr
            }

            if pd.notna(row['value']):
                event_obj["v"] = str(row['value'])[:10]

            if pd.notna(row['error_code']) and False:
                event_obj["e"] = str(row['error_code'])

            lines.append(event_obj)

        lines = json.dumps({"events": lines}, separators=(",", ":"))
        

    else:
        lines.append("\nSEV      | TYPE   | SOURCE                              | NAME                                | VALUE    |  Daily Counts")
        for _, row in grouped.iterrows():
            sev = str(row['severity'])[:8].ljust(8)
            typ = str(row['type'])[:6].ljust(6)
            src = str(row['source'])[:35].ljust(35)
            name = str(row['name'])[:35].ljust(35)
            val = (str(row['value']) if pd.notna(row['value']) else '').ljust(8)
            arr = "[" + ", ".join([f"{Counter(row['_day']).get(i, 0)}" for i in range(5, -1, -1)]) + "]"
            lines.append(f"{sev} | {typ} | {src} | {name} | {val} | {arr}")

        lines = "\n".join(lines)

    return lines

def balance_dataset(dataset):
    classes = [
        element["messages"][-1]["content"]
        for element in dataset
    ]

    per_class = {label : [] for label in set(classes)}
    for label, element in zip(classes,dataset):
        per_class[label].append(element)

    class_counts = Counter(classes)
    target_count = max(class_counts.values())
    balanced_data = []
    for class_id in set(classes):
        samples = per_class[class_id]
        n_samples = len(samples)

        if n_samples >= target_count:
            balanced_data.extend(samples)
        else:
            n_repeates = target_count // n_samples
            n_extra = target_count % n_samples

            extended_data = samples*n_repeates
            if n_extra > 0:
                extended_data.extend(np.random.choice(samples, n_extra, replace=False).tolist())

            balanced_data.extend(extended_data)
    
    np.random.shuffle(balanced_data)
    return balanced_data

def load_model_with_lora(lora_weights_path):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(lora_weights_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(lora_weights_path, trust_remote_code=True)

    return model, tokenizer, device

def format_chat_for_training(messages, tokenizer):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

def get_train_eval_losses(model_path):
    s = json.load(open(os.path.join(model_path, "trainer_state.json")))
    lh = s["log_history"]
    tr = [(e["step"], e["loss"])      for e in lh if "loss" in e]
    ev = [(e["step"], e["eval_loss"]) for e in lh if "eval_loss" in e]
    return tr, ev

import json
def get_references_from_folder(folder):
    references = []
    with open(folder, "r") as f:
        for line in f:
            references.append(json.loads(line)["target"])
    
    return references

def get_predictions_from_folder(folder, classes_names):
    predictions = []
    with open(folder, "r") as f:
        for line in f:
            pred = json.loads(line.strip())["generated"]
            pred = next((kw for kw in classes_names if kw in pred), "")
            predictions.append(pred)
    
    return predictions

def get_f1_metrics_from_folders(main_folder, list_folder, references, classes_names, average="macro"):
    from sklearn.metrics import f1_score

    steps = []
    f1_scores = []
    counts = []
    for name in list_folder:
        steps.append(int(name.split("step_")[1][:-6]))
        curr_folder = os.path.join(main_folder, name)
        predictions = get_predictions_from_folder(curr_folder, classes_names)
        counts.append(Counter(predictions))
        f1_scores.append(f1_score(predictions, references, average=average))
    return steps, [f1_scores, counts]

def tokenize_dataset(formatted_data, tokenizer, max_length=2048*2*2):
    from datasets import Dataset


    def tokenize_function(examples):
        input_ids_batch, attn_batch, labels_batch = [], [], []
        for messages in examples["messages"]:
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            full = tokenizer(full_text, truncation=True, max_length=max_length)
            prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"])
            labels = list(full["input_ids"])
            labels[:prompt_len] = [-100] * min(prompt_len, len(labels))  

            input_ids_batch.append(full["input_ids"])
            attn_batch.append(full["attention_mask"])
            labels_batch.append(labels)

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "labels": labels_batch
        }

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized

def tokenize_dataset_w_issue(formatted_data, tokenizer, max_length=2048*2*2, max_output_length=1000, label_loss_weight=10.0):
    from datasets import Dataset

    def tokenize_function(examples):
        input_ids_batch, attn_batch, labels_batch, weights_batch = [], [], [], []
        truncation_warnings = []

        for idx, item in enumerate(zip(examples["messages"], examples.get("label", [None]*len(examples["messages"])))):
            messages, label_text = item if isinstance(item, tuple) else (item, None)

            # First, get the response length to reserve space for it
            response_text = messages[-1]["content"]
            response_tokens = tokenizer.encode(response_text, add_special_tokens=False)
            response_length = len(response_tokens)

            # Check if response alone exceeds max_output_length
            if response_length > max_output_length:
                truncation_warnings.append(f"Sample {idx}: Response ({response_length} tokens) exceeds max_output_length ({max_output_length})")

            # Calculate how much space we have for the prompt
            # Reserve space for: response + special tokens + small buffer
            reserved_for_output = min(response_length, max_output_length) + 50  # 50 token buffer for special tokens
            max_prompt_length = max_length - reserved_for_output

            # Tokenize prompt with truncation
            prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            prompt_tokens_full = tokenizer.encode(prompt_text, add_special_tokens=False)

            if len(prompt_tokens_full) > max_prompt_length:
                truncation_warnings.append(f"Sample {idx}: Prompt truncated from {len(prompt_tokens_full)} to {max_prompt_length} tokens")

            # Now tokenize the full conversation with max_length
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            full = tokenizer(full_text, truncation=True, max_length=max_length)

            # Get actual prompt length (after chat template and potential truncation)
            prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_prompt_length)["input_ids"])

            # Create labels (mask prompt with -100)
            labels = list(full["input_ids"])
            labels[:prompt_len] = [-100] * min(prompt_len, len(labels))

            # Create token weights for weighted loss
            weights = [0.0] * len(full["input_ids"])  # Start with zeros (no loss on prompt)

            if label_text is not None:
                # Find label token boundaries
                label_tokens = tokenizer.encode(label_text, add_special_tokens=False)
                label_len = len(label_tokens)

                # Label tokens get high weight
                label_start = prompt_len
                label_end = min(label_start + label_len, len(weights))
                for i in range(label_start, label_end):
                    weights[i] = label_loss_weight  # e.g., 10.0

                # Summary tokens get normal weight
                for i in range(label_end, len(weights)):
                    weights[i] = 1.0
            else:
                # No label separation - all response tokens get equal weight
                for i in range(prompt_len, len(weights)):
                    weights[i] = 1.0

            input_ids_batch.append(full["input_ids"])
            attn_batch.append(full["attention_mask"])
            labels_batch.append(labels)
            weights_batch.append(weights)

        # Log truncation warnings
        if truncation_warnings:
            print(f"\nTruncation warnings ({len(truncation_warnings)} samples):")
            for warning in truncation_warnings[:5]:  # Show first 5
                print(f"  {warning}")
            if len(truncation_warnings) > 5:
                print(f"  ... and {len(truncation_warnings) - 5} more")

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "labels": labels_batch,
            "token_weights": weights_batch
        }

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(formatted_data)

    # Tokenize
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized