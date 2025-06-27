import re
import os

# --- Configuration ---
CACHE_DIR = "/run/surfdrive_data/"
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
os.makedirs(CACHE_DIR, exist_ok=True)
print(f"Hugging Face cache directory is configured to: {CACHE_DIR}")

import shutil
import glob
import time
import logging
import wandb
import random
from tqdm import tqdm
import torch
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    Phi3ForCausalLM,
    Phi3Config,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from torch import no_grad

# ACCELERATE: Import Accelerator and helper classes
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed

# --- Data Configuration ---
DATA_PATHS = {
    #"domain_specific": 'data/ELM_2_Whole_excl_thesis_split_clean_2',
    "wiki_en": '/run/surfdrive_data/filtered_articles_en',
    "wiki_nl": '/run/surfdrive_data/filtered_articles_nl',
    "gutenberg": '/run/surfdrive_data/clean',
    "pes2o": '/run/surfdrive_data/filtered_texts'
}
TOKENIZER_PATH = "tokenizer/SHARE_token"

# --- Model & Training Configuration ---
CHECKPOINT_BASE_PATH = "Models/SHARE_4K_Pretrain_Checkpoints"
FINAL_MODEL_PATH = "Models/SHARE_4K_Pretrain_Final"
LEARNING_RATE = 1e-4
# Using small batch size with gradient accumulation to manage memory with padding
BATCH_SIZE = 8 # Per GPU
GRADIENT_ACCUMULATION_STEPS = 1
NUM_EPOCHS = 2
# MODIFIED: Added NUM_WORKERS for parallel data loading
NUM_WORKERS = 8
WEIGHT_DECAY = 0.01
GRADIENT_CLIPPING_NORM = 1.0
SCHEDULER_TYPE = "cosine"
NUM_WARMUP_STEPS = 1000
MAX_LENGTH = 4096
VALIDATION_SET_SIZE = 500
SHUFFLE_BUFFER_SIZE = 10000

# --- Test Mode Configuration ---
TEST_MODE = False
TEST_MODE_SUBSET_SIZE = 1000
TEST_MODE_VALIDATION_SIZE = 100

# --- Stability Enhancements ---
REPEATED_NGRAM_SIZE_RANGE = (1, 13)
REPEATED_NGRAM_THRESHOLD = 16
NO_DECAY_ON_EMBEDDINGS_AND_NORMS = True

# --- Logging & Evaluation ---
WANDB_PROJECT = 'SHARE_4B_Pretrain_Scratch'
WANDB_RUN_NAME = f'SHARE_4B_pretrain_accelerate_{time.strftime("%Y%m%d")}'
EVALUATION_INTERVAL_STEPS = 2500
SAMPLE_GENERATION_INTERVAL_STEPS = 250
PROMPT_TEXT_FOR_SAMPLING = "The foundations of social cohesion are"
LOG_FILE_NAME = f'training_log_pretrain_accelerate_{time.strftime("%Y%m%d_%H%M%S")}.log'

# --- Checkpointing ---
# Note: Manual checkpoint management might be needed without ProjectConfiguration
MAX_CHECKPOINTS_TO_KEEP = 3 

# --- System Setup ---
set_seed(42)


def setup_logging(accelerator):
    """Sets up logging. Only the main process will write to the file."""
    if accelerator.is_main_process:
        logging.basicConfig(filename=LOG_FILE_NAME, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        accelerator.print(f"Logging training info to: {LOG_FILE_NAME}")
        logging.info(f"Using device: {accelerator.device}")
        logging.info(f"Distributed training type: {accelerator.distributed_type}")
        if accelerator.distributed_type == DistributedType.MULTI_GPU:
             logging.info(f"Number of GPUs: {accelerator.num_processes}")
        if accelerator.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
            accelerator.print("Warning: bfloat16 is configured but not supported by the current CUDA device.")
            logging.warning("Warning: bfloat16 is configured but not supported by the current CUDA device.")

def get_data_files_from_path(path, extension=".txt"):
    """Recursively finds all files with a given extension in a directory."""
    if not os.path.isdir(path):
        # Log this, but don't raise an error, to allow for optional data paths
        logging.warning(f"Data path not found or not a directory, skipping: {path}")
        return []
    return glob.glob(os.path.join(path, '**', f'*{extension}'), recursive=True)

def has_excessive_repeated_ngrams(example):
    """Filter function to remove documents with excessive n-gram repetition."""
    tokens = example['input_ids']
    min_n, max_n = REPEATED_NGRAM_SIZE_RANGE
    threshold = REPEATED_NGRAM_THRESHOLD
    doc_len = len(tokens)
    for n in range(min_n, max_n + 1):
        if doc_len < n * threshold: continue
        for i in range(doc_len - n * threshold + 1):
            ngram = tokens[i : i + n]
            if all(tokens[i + j * n : i + (j + 1) * n] == ngram for j in range(1, threshold)):
                return False
    return True

def evaluate_model_performance(model, eval_dataloader, accelerator):
    """Evaluates the model on the evaluation set and returns loss and perplexity."""
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_main_process, leave=False):
            outputs = model(**batch)
            loss = outputs.loss
            # Gathers the per-batch loss from all processes
            losses.append(accelerator.gather_for_metrics(loss.unsqueeze(0)))
    
    losses = torch.cat(losses)
    
    try:
        avg_loss = torch.mean(losses).item()
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
    except OverflowError:
        avg_loss, perplexity = float("inf"), float("inf")
    model.train()
    return avg_loss, perplexity

def main():
    # Removed split_batches=True as we are reverting to a fixed-padding strategy.
    accelerator = Accelerator(
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, 
        log_with="wandb",
        mixed_precision='bf16'
    )
    setup_logging(accelerator)

    # --- 1) Load Tokenizer ---
    accelerator.print(f"Loading tokenizer from: {TOKENIZER_PATH}")
    if not os.path.exists(TOKENIZER_PATH): raise FileNotFoundError(f"Tokenizer path not found: {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_cache=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # --- 2) Set up data paths and parameters ---
    if TEST_MODE:
        accelerator.print("\n--- RUNNING IN TEST MODE ---")
        active_data_paths, current_validation_size = {"domain_specific": DATA_PATHS["domain_specific"]}, TEST_MODE_VALIDATION_SIZE
    else:
        active_data_paths = DATA_PATHS
        current_validation_size = VALIDATION_SET_SIZE
    
    # --- 3) Load and Preprocess Datasets ---
    accelerator.print("Starting dataset preparation...")

    files_to_load_list = [None]
    if accelerator.is_main_process:
        accelerator.print("Main process is discovering data files...")
        all_files = []
        for name, path in active_data_paths.items():
            accelerator.print(f"- Searching in '{name}' at path: {path}")
            discovered_files = get_data_files_from_path(path)
            all_files.extend(discovered_files)
            accelerator.print(f"  ... found {len(discovered_files)} files.")
        
        files_to_load_list[0] = all_files
        accelerator.print(f"Total files found by main process: {len(all_files)}")

    if accelerator.distributed_type != DistributedType.NO:
        torch.distributed.broadcast_object_list(files_to_load_list, src=0)

    accelerator.wait_for_everyone()

    files_to_load = files_to_load_list[0]
    if not files_to_load:
        if accelerator.is_main_process:
            raise ValueError("No data files were found. Aborting.")
        else:
            accelerator.wait_for_everyone()
            return

    num_training_docs = len(files_to_load) if not TEST_MODE else TEST_MODE_SUBSET_SIZE
    accelerator.print(f"All processes will now load a dataset from {len(files_to_load)} files.")

    combined_dataset = load_dataset("text", data_files={"train": files_to_load}, streaming=True, split="train")

    if TEST_MODE: 
        combined_dataset = combined_dataset.take(TEST_MODE_SUBSET_SIZE)

    def preprocess(ex):
        ex['text'] = '\n'.join([l.lower() if l.isupper() else l for l in ex['text'].split('\n')])
        return ex

    preprocessed_dataset = combined_dataset.map(preprocess).shuffle(seed=42, buffer_size=SHUFFLE_BUFFER_SIZE)
    eval_dataset = preprocessed_dataset.take(current_validation_size)
    train_dataset = preprocessed_dataset.skip(current_validation_size)

    # Reverting to padding all samples to max_length to guarantee uniform tensor sizes.
    def tokenize(ex): 
        return tokenizer(ex["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")

    tokenized_train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text']).filter(has_excessive_repeated_ngrams)
    tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=['text'])

    # --- 4) Calculate Training Steps ---
    effective_batch_size = BATCH_SIZE * accelerator.num_processes * GRADIENT_ACCUMULATION_STEPS
    if not TEST_MODE:
        estimated_num_docs = 11_547_071 
        steps_per_epoch = estimated_num_docs // effective_batch_size
    else:
        steps_per_epoch = (TEST_MODE_SUBSET_SIZE - current_validation_size) // effective_batch_size

    total_training_steps = steps_per_epoch * NUM_EPOCHS
    accelerator.print(f"Estimated total training steps: {total_training_steps}, Effective batch size: {effective_batch_size}")


    # --- 5) Set up the model ---
    accelerator.print("Initializing new model from scratch...")
    model_config = Phi3Config(
        use_cache=False, 
        vocab_size=len(tokenizer), 
        bos_token_id=tokenizer.bos_token_id, 
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.pad_token_id, 
        attn_implementation="flash_attention_2" if accelerator.device.type == 'cuda' else 'eager',
        torch_dtype=torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto'
    )
    model = Phi3ForCausalLM(model_config)
    
    if accelerator.mixed_precision == 'bf16':
        model = model.to(torch.bfloat16)

    model.gradient_checkpointing_enable()
    
    # --- 6) Data Collator & Dataloaders ---
    # Reverting to the standard data collator.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # MODIFIED: Added num_workers and pin_memory for faster data loading.
    train_dataloader = DataLoader(
        tokenized_train_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        tokenized_eval_dataset, 
        batch_size=BATCH_SIZE, 
        collate_fn=data_collator, 
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    accelerator.print(f"Validation will be performed on ~{current_validation_size} samples.")

    # --- 7) Optimizer and Scheduler ---
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if param.ndim == 1 or "norm" in name or "embed" in name: no_decay_params.append(param)
        else: decay_params.append(param)
    optimizer_grouped_parameters = [{'params': decay_params, 'weight_decay': WEIGHT_DECAY}, {'params': no_decay_params, 'weight_decay': 0.0}]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters if NO_DECAY_ON_EMBEDDINGS_AND_NORMS else model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = get_scheduler(SCHEDULER_TYPE, optimizer=optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=total_training_steps)

    # --- 8) Prepare with Accelerator and Initialize Tracker ---
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader, lr_scheduler)
    
    if accelerator.is_main_process:
        accelerator.init_trackers(WANDB_PROJECT, config={
            "learning_rate": LEARNING_RATE, "per_gpu_batch_size": BATCH_SIZE, "total_batch_size": effective_batch_size,
            "num_epochs": NUM_EPOCHS, "weight_decay": WEIGHT_DECAY, "max_length": MAX_LENGTH,
            "num_gpus": accelerator.num_processes, "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "total_training_steps": total_training_steps, "test_mode": TEST_MODE, "mixed_precision": accelerator.mixed_precision,
            "num_workers": NUM_WORKERS
        }, init_kwargs={"wandb": {"name": WANDB_RUN_NAME + ("_TEST" if TEST_MODE else "")}})

    # --- 9) Resume from Checkpoint if available ---
    completed_steps = 0
    resume_from_checkpoint = None
    
    if os.path.isdir(CHECKPOINT_BASE_PATH):
        checkpoint_dirs = glob.glob(os.path.join(CHECKPOINT_BASE_PATH, "checkpoint_*"))
        if checkpoint_dirs:
            latest_checkpoint_dir = max(checkpoint_dirs, key=os.path.getctime)
            match = re.search(r"checkpoint_(\d+)", latest_checkpoint_dir)
            if match:
                resume_from_checkpoint = latest_checkpoint_dir
                completed_steps = int(match.group(1))
                accelerator.print(f"Resuming training from checkpoint: {resume_from_checkpoint} at step {completed_steps}")

    if resume_from_checkpoint:
        try:
            accelerator.load_state(resume_from_checkpoint)
        except Exception as e:
            accelerator.print(f"Could not load state from {resume_from_checkpoint}. Starting from scratch. Error: {e}")
            completed_steps = 0 

    # --- 10) Training Loop ---
    accelerator.print(f"Starting Training from step {completed_steps}...")
    progress_bar = tqdm(total=total_training_steps, initial=completed_steps, disable=not accelerator.is_main_process)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch in train_dataloader:
            if completed_steps >= total_training_steps:
                break
            
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING_NORM)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                lr_scheduler.step()
                completed_steps += 1
                
                accelerator.log({"train_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}, step=completed_steps)
                
                if completed_steps > 0 and (completed_steps % EVALUATION_INTERVAL_STEPS == 0 or completed_steps == total_training_steps):
                    accelerator.print(f"\n--- Starting Evaluation @ Step {completed_steps} ---")
                    eval_loss, perplexity = evaluate_model_performance(model, eval_dataloader, accelerator)
                    accelerator.log({"eval_loss": eval_loss, "perplexity": perplexity}, step=completed_steps)
                    accelerator.print(f"Validation | Step {completed_steps} | Loss: {eval_loss:.4f} | Perplexity: {perplexity:.4f}")
                    
                    accelerator.wait_for_everyone()
                    output_dir = os.path.join(CHECKPOINT_BASE_PATH, f"checkpoint_{completed_steps}")
                    accelerator.save_state(output_dir)
                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(output_dir)

                if accelerator.is_main_process and completed_steps > 0 and completed_steps % SAMPLE_GENERATION_INTERVAL_STEPS == 0:
                    model.eval()
                    unwrapped_model = accelerator.unwrap_model(model)
                    inputs = {k: v.to(accelerator.device) for k, v in tokenizer(PROMPT_TEXT_FOR_SAMPLING, return_tensors="pt").items()}
                    inputs.pop("token_type_ids", None)
                    generated_ids = unwrapped_model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.7, top_k=50, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
                    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    accelerator.print(f"\nSample Text @ {completed_steps}: {generated_text}")
                    accelerator.log({"sample_text": wandb.Html(f"<pre>{generated_text}</pre>")}, step=completed_steps)
                    model.train()
            
        if completed_steps >= total_training_steps:
            break

    progress_bar.close()

    # --- 11) Final Model Saving ---
    if not TEST_MODE:
        accelerator.wait_for_everyone()
        accelerator.print("\nTraining complete. Saving final model...")
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(FINAL_MODEL_PATH, safe_serialization=True)
            tokenizer.save_pretrained(FINAL_MODEL_PATH)
            accelerator.print(f"Final model saved to {FINAL_MODEL_PATH}")

    accelerator.end_training()
    accelerator.print("Script finished.")

if __name__ == "__main__":
    main()
