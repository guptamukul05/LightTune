"""
src/data/preprocess.py
Loads the dataset named in training_config.yaml, formats Alpaca-style prompts,
tokenizes with the model's tokenizer, and saves train/val splits to data/train and data/val.
"""
import argparse
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer

ALPACA_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that provides "
    "further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
)
ALPACA_PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that appropriately "
    "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n{output}"
)


def read_configs(model_cfg_path, train_cfg_path):
    with open(model_cfg_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(train_cfg_path) as f:
        train_cfg = yaml.safe_load(f)
    return model_cfg, train_cfg


def format_example(example):
    if example.get("input", "").strip():
        text = ALPACA_PROMPT.format(**example)
    else:
        text = ALPACA_PROMPT_NO_INPUT.format(**example)
    return {"text": text}


def tokenize_fn(tokenizer, max_seq_length):
    def _fn(example):
        out = tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out
    return _fn


def main(model_cfg_path, train_cfg_path):
    model_cfg, train_cfg = read_configs(model_cfg_path, train_cfg_path)
    d_cfg = train_cfg["dataset"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model"]["hf_repo"],
        trust_remote_code=model_cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(d_cfg["hf_repo"], split=d_cfg["train_split"])
    max_samples = d_cfg.get("max_samples")
    if max_samples:
        raw = raw.shuffle(seed=d_cfg["seed"]).select(range(min(max_samples, len(raw))))
    raw = raw.map(
        format_example,
        remove_columns=[c for c in raw.column_names if c != "text"],
    )

    tokenize = tokenize_fn(tokenizer, model_cfg["model"]["max_seq_length"])
    tokenized = raw.map(tokenize, remove_columns=["text"])

    split = tokenized.train_test_split(test_size=d_cfg["val_ratio"], seed=d_cfg["seed"])
    split["train"].save_to_disk("data/train")
    split["test"].save_to_disk("data/val")

    print(f"train examples: {len(split['train'])}")
    print(f"val examples:   {len(split['test'])}")
    print("Saved to data/train and data/val")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config", default="configs/model_config.yaml")
    parser.add_argument("--train_config", default="configs/training_config.yaml")
    args = parser.parse_args()
    main(args.model_config, args.train_config)
