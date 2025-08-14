# train_lora.py
import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_dataset(train_file: str, tokenizer, max_length: int = 512):
    print(f"[INFO] Loading dataset from {train_file} ...")
    ds = load_dataset("json", data_files={"train": train_file})["train"]

    def make_text(example):
        prompt = example.get("prompt", "").strip()
        completion = example.get("completion", "").strip()
        # Разделитель между prompt и completion; добавляем eos, если есть
        eos = tokenizer.eos_token if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None else ""
        text = prompt + "\n\n###\n\n" + completion + eos
        return {"text": text}

    # Преобразуем в поле text
    ds = ds.map(make_text, remove_columns=[c for c in ds.column_names if c in ("prompt", "completion")], batched=False)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length)

    ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    return ds

def train_lora(
    base_model: str,
    train_file: str,
    lora_dir: str,
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 2e-4,
    seed: int = 42,
    max_length: int = 512,
    device_map: str = "auto",
    force: bool = False
):
    set_seed(seed)

    lora_path = Path(lora_dir)
    if lora_path.exists() and any(lora_path.iterdir()) and not force:
        print(f"[INFO] LORA adapter already exists at {lora_path}. Skipping training (use --force to override).")
        info = {"status": "skipped", "lora_dir": str(lora_path.resolve())}
        with open(lora_path / "training_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return

    # Ensure output dir exists
    lora_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading tokenizer and base model: {base_model} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Prepare dataset
    ds = prepare_dataset(train_file, tokenizer, max_length=max_length)

    print("[INFO] Loading base model (this may use significant memory)...")
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map=device_map, torch_dtype=torch_dtype)
    model.resize_token_embeddings(len(tokenizer))

    print("[INFO] Applying LoRA adapters...")
    # target_modules may need tuning depending on model architecture
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=str(lora_path),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=data_collator,
    )

    print("[INFO] Starting training...")
    trainer.train()
    print("[INFO] Training finished. Saving adapter...")

    model.save_pretrained(str(lora_path))

    train_info = {
        "base_model": base_model,
        "train_file": train_file,
        "lora_dir": str(lora_path.resolve()),
        "epochs": epochs,
        "batch": batch_size,
        "lr": lr,
        "seed": seed
    }
    with open(lora_path / "training_info.json", "w", encoding="utf-8") as f:
        json.dump(train_info, f, ensure_ascii=False, indent=2)

    print(f"[INFO] LoRA adapter saved to {lora_path}")

def parse_args():
    p = argparse.ArgumentParser(description="Train LoRA adapter from chat dataset")
    p.add_argument("--force", action="store_true", help="Force retrain and overwrite existing adapter")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # --------------------- ЯВНО ЗАДАЙТЕ ЗДЕСЬ СВОИ ПУТИ / ПАРАМЕТРЫ ---------------------
    # Пропишите абсолютный или относительный путь к вашему jsonl (prompt/completion)
    TRAIN_FILE = r"D:\projects\python\TelegramGraphNet\TeleNet\dataset\chat_759575591_messages_dataset.jsonl"      # <- Путь к файлу .jsonl с парами prompt/completion
    BASE_MODEL = "gpt2"                   # <- Базовая модель (можно сменить)
    LORA_DIR = r"../lora_out"  # <- Куда сохранить adapter
    EPOCHS = 3
    BATCH = 4
    LR = 2e-4
    SEED = 42
    MAX_LENGTH = 512
    DEVICE_MAP = "auto"
    # ------------------------------------------------------------------------------

    tf_path = Path(TRAIN_FILE)
    if not tf_path.exists():
        print(f"[ERROR] Train file not found: {tf_path.resolve()}")
        print("Укажите корректный путь в блоке if __name__ == '__main__'")
        raise SystemExit(1)

    train_lora(
        base_model=BASE_MODEL,
        train_file=str(tf_path),
        lora_dir=LORA_DIR,
        epochs=EPOCHS,
        batch_size=BATCH,
        lr=LR,
        seed=SEED,
        max_length=MAX_LENGTH,
        device_map=DEVICE_MAP,
        force=args.force
    )
