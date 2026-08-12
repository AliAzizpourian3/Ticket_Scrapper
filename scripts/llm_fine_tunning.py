import requests
from datetime import datetime
from collections import Counter, defaultdict
import pandas as pd
import numpy as np
import json
import sys
sys.path.append("../src")
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import pickle
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import copy
import sklearn as sk
from dataclasses import dataclass

from tqdm import tqdm

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM, PeftModel

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-360M-Instruct",
    local_files_only=False,
    resume_download=True,
    headers={"X-Skip-Verify": "true"}
)

DATABASE_DIR = "../Database"
TICKET_DIR = os.path.join(DATABASE_DIR, "Ticket_Extraction")
MAX_LENGTH = 2048 * 2 * 2

# Prepare for training

def load_model_and_tokenizer(model_name):
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules, 
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,  # load in float32; Trainer handles fp16 casting
        trust_remote_code=False,
        attn_implementation="eager"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)


    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(f"Model loaded: {model.num_parameters():,} parameters")

    return model, tokenizer, device

@dataclass
class CustomDataCollator:
    """
    Data collator that handles token_weights along with standard fields.
    """
    tokenizer: AutoTokenizer
    padding: bool = True

    def __call__(self, features):
        # Separate token_weights from other features
        token_weights = [f.get("token_weights") for f in features]
        features_copy = [{k: v for k, v in f.items() if k != "token_weights"} for f in features]


        # Use standard collator for other fields
        from transformers.data.data_collator import DataCollatorForSeq2Seq
        standard_collator = DataCollatorForSeq2Seq(self.tokenizer, padding=self.padding)
        batch = standard_collator(features_copy)

        # Manually pad token_weights if present
        if any(w is not None for w in token_weights):
            # Get max length from batch
            max_length = batch["input_ids"].shape[1]

            # Pad token_weights to max_length
            padded_weights = []
            for weights in token_weights:
                if weights is None:
                    # If no weights provided, default to all 1.0
                    padded_weights.append([1.0] * max_length)
                else:
                    # Pad with 0.0 (no loss on padding)
                    padded = weights + [0.0] * (max_length - len(weights))
                    padded_weights.append(padded)

            batch["token_weights"] = torch.tensor(padded_weights, dtype=torch.float32, device=batch["input_ids"].device)

        return batch


class CustomTrainer(Trainer):

    def __init__(self, tokenizer, eval_dataset, eval_raw, test_raw, output_dir, max_new_tokens=15, batch_size=4, label_loss_weight=10.0, *args, **kwargs):
        super().__init__(eval_dataset=eval_dataset,*args, **kwargs)
        self.tokenizer = tokenizer
        self.label_loss_weight = label_loss_weight

        self.eval_predictions_dir = os.path.join(output_dir, "predictions", "eval")
        os.makedirs(self.eval_predictions_dir, exist_ok=True)
        self.test_predictions_dir = os.path.join(output_dir, "predictions", "test")
        self.last_eval_step = -1
        self.current_eval_file = None
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        self.eval_dataset_raw = self.prepare_raw_datasets(eval_raw, self.eval_predictions_dir)

        self.test_dataset_raw = self.prepare_raw_datasets(test_raw, self.test_predictions_dir)

    def prepare_raw_datasets(self, dataset, path_file): 
        messages = [raw["messages"] for raw in dataset]
        os.makedirs(path_file, exist_ok=True)
        with open(os.path.join( path_file, "references.jsonl"), "w") as f:
            for mes in messages:
                json.dump({"target": mes[-1]["content"]}, f)
                f.write("\n")
        messages =[mes[:-1] for mes in messages]
        prompts = [
            self.tokenizer.apply_chat_template(mes, tokenize=False, add_generation_prompt=True) 
            for mes in messages]
        prompts = [
            self.tokenizer(prompt, return_tensors="pt").to(self.args.device)
            for prompt in prompts
        ]
        return prompts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss function that applies token-level weighting.
        Higher weight for label tokens, normal weight for summary tokens.
        """
        labels = inputs.get("labels")
        token_weights = inputs.get("token_weights")  

        # Forward pass
        outputs = model(**inputs)

        if token_weights is not None:
            # Compute per-token cross-entropy loss
            logits = outputs.logits

            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = token_weights[..., 1:].contiguous()

            # Compute loss per token (reduction='none' gives per-token loss)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )

            # Apply token weights
            weighted_loss = loss * shift_weights.view(-1)

            # Mask out -100 (padding/prompt tokens)
            mask = (shift_labels.view(-1) != -100)
            final_loss = (weighted_loss * mask).sum() / mask.sum()

            # Optional: log label vs summary loss separately
            if self.state.global_step % self.args.logging_steps == 0:
                label_mask = (shift_weights.view(-1) >= self.label_loss_weight * 0.9) & mask
                summary_mask = (shift_weights.view(-1) <= 1.1) & (shift_weights.view(-1) >= 0.9) & mask

                if label_mask.sum() > 0 and summary_mask.sum() > 0:
                    label_loss = (weighted_loss * label_mask).sum() / label_mask.sum()
                    summary_loss = (loss * summary_mask).sum() / summary_mask.sum()
                    print(f"  [Step {self.state.global_step}] Label loss: {label_loss.item():.4f}, Summary loss: {summary_loss.item():.4f}")

            return (final_loss, outputs) if return_outputs else final_loss
        else:
            # Fall back to default loss if no weights provided
            return (outputs.loss, outputs) if return_outputs else outputs.loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        ) 


        self.run_generation_metrics(self.eval_dataset_raw, self.eval_predictions_dir, self.batch_size)

        # self.run_generation_metrics(self.test_dataset_raw, self.test_predictions_dir, self.batch_size)

        return metrics


    @torch.no_grad()
    def run_generation_metrics(self, dataset, path_file, batch_size):
        model = self.model
        model.eval()

        all_predictions = []

        for index in tqdm(range(0, len(dataset), batch_size), desc="Generating for metrics"):
            batch = dataset[index:index+batch_size]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [b["input_ids"].squeeze(0).flip(0) for b in batch],
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id
            ).flip(dims=[1]).to(self.args.device)

            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [b["attention_mask"].squeeze(0).flip(0) for b in batch],
                batch_first=True,
                padding_value=0
            ).flip(dims=[1]).to(self.args.device)


            outputs = model.generate(
              input_ids=input_ids,
              attention_mask=attention_mask,
              max_new_tokens=self.max_new_tokens,
              eos_token_id=self.tokenizer.eos_token_id,
              pad_token_id=self.tokenizer.pad_token_id,
            )
            for j, output in enumerate(outputs):
                # original_len = batch[j]["input_ids"].shape[1]
                new_tokens = self.tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True).strip()
                all_predictions.append(new_tokens)


        eval_file = os.path.join(
            path_file,
            f"step_{self.state.global_step:06d}.jsonl"
        )
        with open(eval_file, "w") as f:
            for pred in all_predictions:
                json.dump({"generated": pred}, f)
                f.write("\n")
        model.train()

def train_model(model, tokenizer, train_data, val_data, val_raw, test_raw, output_dir, num_epochs=6, batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-5, eval_steps=10, save_steps=10):
    """Fine-tune the model."""

    # Training arguments
    use_cuda = torch.cuda.is_available()
    model.config.use_cache = False  
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir='./logs',
        logging_steps=10,
        report_to="none",
        fp16=use_cuda,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    # Data collator
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False
    # )
    # data_collator = SmolLMDataCollator(tokenizer=tokenizer)
    
    # Change depending on no issue / issue
    # data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True) # No issue
    data_collator = CustomDataCollator(tokenizer=tokenizer, padding=True)       # Using issue

    # Initialize Trainer
    # trainer = Trainer(
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        eval_raw=val_raw,
        test_raw=test_raw,
        data_collator=data_collator,
        output_dir = output_dir,
        batch_size=batch_size,
    )

    print("Starting fine-tuning...")
    import glob
    checkpoints = glob.glob(f"{output_dir}/checkpoint-*")
    if checkpoints:
        latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        print(f"Resuming from {latest_checkpoint}")
        trainer.train(resume_from_checkpoint=latest_checkpoint)
    else:
        print("Starting fresh training")
        trainer.train()
    # trainer.train()

    # Save final model
    final_output = f"{output_dir}_final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    # print(f"Training complete! Model saved to {final_output}")

    return trainer

def train_main():
    from model_training.llm_utils import balance_dataset, tokenize_dataset, tokenize_dataset_w_issue
    import sklearn as sk
    
    model_name = "microsoft/Phi-3-mini-4k-instruct"  # Change to "meta-llama/Llama-3.2-3B-Instruct" if needed
    # model_name = "microsoft/Phi-3-mini-4k-instruct-gguf"
    model_name = "HuggingFaceTB/SmolLM2-360M-Instruct"
    model, tokenizer, device = load_model_and_tokenizer(model_name)

    format_dir = os.path.join(TICKET_DIR, "formatted_data_issue.pkl")
    with open(format_dir, "rb") as f:
        formatted_data = pickle.load(f)


    indexes = np.arange(len(formatted_data))
    train_val_indexes, test_indexes = sk.model_selection.train_test_split(indexes, test_size=0.2, random_state=42)
    train_indexes, val_indexes = sk.model_selection.train_test_split(train_val_indexes, test_size=0.15, random_state=42)
    train = [formatted_data[i] for i in train_indexes]
    val = [formatted_data[i] for i in val_indexes]
    test = [formatted_data[i] for i in test_indexes]

    
    train_balanced = balance_dataset(train)

    DIFFERENT_TOKENIZATION = 1
    if DIFFERENT_TOKENIZATION:
        train_tokenized = tokenize_dataset_w_issue(train_balanced, tokenizer, label_loss_weight=10.0)
        val_tokenized = tokenize_dataset_w_issue(val, tokenizer, label_loss_weight=10.0)
    else:
        train_tokenized = tokenize_dataset(train_balanced, tokenizer)
        val_tokenized = tokenize_dataset(val, tokenizer)

    OUTPUT_DIR = os.path.join(TICKET_DIR, "LLM_models", "model_-1_cluster_copy")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    

    trainer = train_model(model, tokenizer, train_tokenized, val_tokenized, val, test, OUTPUT_DIR, gradient_accumulation_steps=8, batch_size=4, eval_steps=10, save_steps=10)

def format_messages(ticket_dir, JUST_LABEL=True, NUMERICAL_CLASS=False):
    from model_training.model_utils import class_short_names, new_class_short_names


    print("This is the ticket Directory: ", ticket_dir)
    summaries = []
    names = []
    charger_id = []
    all_tickets   = os.path.join(ticket_dir, "Tickets_Trimmed_Summary")
    for index, file_name in tqdm(enumerate(os.listdir(all_tickets))):
        names.append(file_name)
        with open(os.path.join(all_tickets, file_name), "r") as in_file:
            summaries.append(in_file.read())
        json_load = json.load(open(os.path.join(ticket_dir, "Tickets_Trimmed", file_name.split(".")[0]+".json"), "r"))
        charger_id.append([json_load["chargerID"], json_load["created_on"], json_load["incident_id"], int(index)])
    

    charger_id = np.array(charger_id)

    model_name = "allenai-specter"
    k_means_labels = np.load(os.path.join(ticket_dir, "Ticket_Embeddings", f"{model_name}_k_means_labels.npy"))

    specific_tickets = charger_id
    per_charger = defaultdict(list)
    for el in specific_tickets:
        per_charger[el[0]].append([pd.Timestamp(str(el[1])), el[2], int(el[3])])


    classes_names = list(class_short_names.values())

    input_dir = os.path.join(ticket_dir, "Queries")

    system_message = (
        "You are an expert at analyzing electric vehicle charger event logs. "
        "Given event log data from the week before a maintenance ticket was submitted, "
        "classify the fault type" + (". " if JUST_LABEL else " and summarize the issue. ") +

        f"Available Fault Classes: \n{chr(10).join(f'- {name}' for name in classes_names)}\n\n"
        "Just respond about the class type, and in JSON format. Don't provide explanations or reasoning."
    )


    formatted_data = []
    for charger, val in per_charger.items():

        for ts, incident, index in val:
            query_input = os.path.join(input_dir, f"{incident}.txt")
            if not os.path.exists(query_input):
                continue

            with open(query_input, "r") as in_file:
                query = in_file.read()

            issue = summaries[index].split("\n")[1]
            ticket_id = names[index].split(".")[0]
            class_ = new_class_short_names[k_means_labels[index]]
            numerical_class_ = k_means_labels[index]

            used_class = numerical_class_ if NUMERICAL_CLASS else class_

            label_text = f'{used_class}"}}'
            full_text = label_text  + f'\n\n Summary: {issue}'

            final_text = label_text if JUST_LABEL else full_text

            content_text = (f"Based on the following event logs" + ("" if JUST_LABEL else ", provide a summary of the issue, and") + f" classify the fault type:\n\n {query} \n\n" + f'Output: {{"class": "')

            messages = [
                {
                    "role": "system", 
                    "content": system_message
                },
                {
                    "role": "user",
                    # "content": f"Based on the following event logs, provide a summary of the issue, and classify the fault type:\n\n {sample['query']}" 
                    "content": content_text
                },
                {
                    "role": "assistant",
                    "content": final_text
                }
            ]

            formatted_data.append({
                "messages" : messages,
                "ticket_id": ticket_id,
                "charger": charger,
                "summary": summaries[int(index)],
                "label_text": label_text
            })

    return formatted_data

def format_messages_main():
    formatted_data = format_messages(TICKET_DIR, JUST_LABEL=1, NUMERICAL_CLASS=0)
    output_dir = os.path.join(TICKET_DIR, "formatted_data.pkl")
    with open(output_dir, "wb") as f:
        pickle.dump(formatted_data, f)



if __name__ == "__main__":

    np.random.seed(42)
    # format_messages_main()
    train_main()


