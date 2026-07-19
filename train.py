"""
Training Script for Farmer Crop Advisory AI
Fine-tunes Qwen 2.5 3B on Indian agricultural data using QLoRA
Usage: python train.py --dataset data/final_dataset_v4.jsonl
"""
import os, json, torch, argparse
os.environ["WANDB_DISABLED"] = "true"

from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

SYSTEM_PROMPT = """आप भारतीय किसानों के लिए एक कृषि विशेषज्ञ सहायक हैं।
आप फसल रोग, कीट प्रबंधन, उर्वरक और खेती की सलाह देते हैं।
हमेशा व्यावहारिक सलाह दें और स्थानीय KVK से पुष्टि करने की सलाह दें।"""


def format_to_chatml(batch):
    texts = []
    for inst, out in zip(batch["instruction"], batch["output"]):
        texts.append(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{inst}<|im_end|>\n"
            f"<|im_start|>assistant\n{out}<|im_end|>"
        )
    return {"text": texts}


def main(args):
    # Load dataset
    records = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line.strip()))
    dataset = Dataset.from_list(records)
    print(f"Dataset: {len(dataset)} examples")

    # Load model in 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B-Instruct",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
        ),
        device_map="auto", torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Add LoRA
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    # Format and tokenize
    dataset_fmt = dataset.map(format_to_chatml, batched=True,
        remove_columns=["instruction", "input", "output"])

    def tokenize(example):
        result = tokenizer(example["text"], truncation=True,
                          max_length=args.max_length, padding=False)
        result["labels"] = result["input_ids"].copy()
        return result

    dataset_tok = dataset_fmt.map(tokenize, batched=True, remove_columns=["text"])
    split = dataset_tok.train_test_split(test_size=0.05, seed=42)
    print(f"Train: {len(split['train'])} | Val: {len(split['test'])}")

    # Train
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            warmup_steps=30, num_train_epochs=args.epochs,
            learning_rate=2e-4, fp16=True, logging_steps=10,
            save_strategy="steps", save_steps=100, save_total_limit=3,
            optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=42,
            report_to="none", gradient_checkpointing=True,
        ),
        train_dataset=split["train"], eval_dataset=None,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final-model")
    tokenizer.save_pretrained(f"{args.output_dir}/final-model")
    print(f"Model saved to {args.output_dir}/final-model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/final_dataset_v4.jsonl")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    main(parser.parse_args())