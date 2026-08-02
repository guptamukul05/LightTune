# ⚡ LightTune

**Parameter-Efficient Fine-Tuning of Small Language Models under GPU Resource Constraints**

A complete benchmarking pipeline that fine-tunes [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) on the Stanford Alpaca instruction dataset using **LoRA** and **QLoRA**, runs on a GPU cluster via Slurm, and produces a detailed side-by-side comparison of training efficiency, GPU memory usage, and model quality across all three configurations.

---

## 📌 Pipeline Overview

| Stage | Description |
|---|---|
| 🗄️ Preprocessing | Downloads Alpaca, formats prompts, tokenises, and saves train/val splits |
| 📊 Baseline Eval | Evaluates the unmodified Qwen2.5-1.5B zero-shot on the validation set |
| 🔵 LoRA Fine-Tuning | Trains only 1.18% of parameters using low-rank adapter matrices |
| 🟣 QLoRA Fine-Tuning | Same as LoRA but base model loaded in 4-bit NF4 quantised format |
| 🚀 Inference Bench | Compares baseline vs fine-tuned latency and throughput via SGLang |
| 📈 Analysis | Aggregates all results into CSV tables and GPU memory plots |

---

## 🧠 What is LoRA?

Full fine-tuning of a 1.5 billion parameter model updates every single weight, which demands tens of gigabytes of GPU memory. LoRA solves this by freezing all original weights and inserting two small trainable matrices **A** and **B** alongside each attention and MLP projection layer.

The effective weight update is:

```
ΔW = A × B    where  A ∈ ℝ^(d×r)  and  B ∈ ℝ^(r×k)  with  r ≪ min(d, k)
```

During the forward pass:

```
h = W₀x + (α / r) × B × A × x
```

With rank `r = 16` and scaling `α = 32`, this project trains only **18.4 million parameters** out of 1.56 billion — just **1.18%** of the total, cutting GPU memory by around 60 to 70 percent compared to full fine-tuning.

---

## 🟣 What is QLoRA?

QLoRA applies the same LoRA adapters but first compresses the base model weights from 16-bit to **4-bit NF4 (Normal Float 4)** format using bitsandbytes. The adapter matrices themselves remain in BF16 precision. Double quantisation is also enabled, which compresses the quantisation constants themselves and saves an additional 0.37 bits per parameter on average.

---

## 🗂️ Repository Structure

```
hpc-peft-project/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
├── configs/
│   ├── model_config.yaml          # Model, LoRA, QLoRA, and precision settings
│   └── training_config.yaml       # Dataset, epochs, batch size, LR, scheduler
├── scripts/
│   ├── setup_env.sh               # Creates virtual environment via uv, installs deps
│   ├── run_training.sh            # Runs preprocessing, training, and evaluation
│   ├── run_baseline.sh            # Zero-shot baseline evaluation only
│   └── entrypoint.sh              # Docker container entrypoint
├── src/
│   ├── data/
│   │   └── preprocess.py          # Dataset download, prompt formatting, tokenisation
│   ├── models/
│   │   └── load_model.py          # Model loader for baseline, LoRA, and QLoRA
│   ├── training/
│   │   ├── train_lora.py          # LoRA fine-tuning script
│   │   └── train_qlora.py         # QLoRA fine-tuning script
│   ├── evaluation/
│   │   └── evaluate.py            # Validation loss, accuracy, F1 evaluation
│   └── utils/
│       └── gpu_monitor.py         # Background GPU memory and utilisation logger
├── data/                          # Preprocessed train/val splits  [git-ignored]
├── checkpoints/                   # Saved LoRA and QLoRA adapter weights  [git-ignored]
├── logs/                          # Training summaries, eval JSONs, GPU CSVs  [git-ignored]
├── exports/                       # Frozen environment snapshots  [git-ignored]
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

**`configs/model_config.yaml`**

```yaml
model:
  name: "Qwen2.5-1.5B"
  hf_repo: "Qwen/Qwen2.5-1.5B"
  max_seq_length: 512
  trust_remote_code: true

lora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: "none"

qlora:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

precision:
  mixed_precision: "bf16"
```

**`configs/training_config.yaml`**

```yaml
dataset:
  hf_repo: "tatsu-lab/alpaca"
  max_samples: 2000
  val_ratio: 0.1
  seed: 42

training:
  num_train_epochs: 2
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 4      # effective batch size = 32
  learning_rate: 2.0e-4
  lr_scheduler_type: "cosine"
  warmup_ratio: 0.03
  weight_decay: 0.01
  gradient_checkpointing: true
  bf16: true
  output_dir: "checkpoints"
  logging_dir: "logs"

resource_monitor:
  interval_seconds: 30
```

---

## 🖥️ Hardware

| Component | Details |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB × 8 |
| GPU Memory per card | 81,920 MB |
| CUDA | 12.3 |
| Driver | 525.147.05 |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/guptamukul05/LightTune.git
cd LightTune
```

### 2. Set up the environment

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

This installs Python 3.11 via `uv`, creates a virtual environment, and installs all dependencies. No sudo or containers required. CUDA availability is verified automatically at the end of the setup.

### 3. Configure your environment variables

```bash
cp .env.example .env
# Edit .env and set your HuggingFace token
nano .env
```

### 4. Preprocess the dataset

```bash
python src/data/preprocess.py \
    --model_config configs/model_config.yaml \
    --train_config configs/training_config.yaml
```

Downloads the Alpaca dataset, selects 2,000 samples, formats each row into the Alpaca prompt template, tokenises with the Qwen2.5 tokeniser, and saves the splits to `data/train` and `data/val`.

### 5. Run baseline evaluation

```bash
bash scripts/run_baseline.sh
```

### 6. Run LoRA or QLoRA training

```bash
# LoRA
python src/training/train_lora.py \
    --model_config configs/model_config.yaml \
    --train_config configs/training_config.yaml

# QLoRA
python src/training/train_qlora.py \
    --model_config configs/model_config.yaml \
    --train_config configs/training_config.yaml
```

### 7. Submit to a Slurm cluster

```bash
sbatch scripts/submit_lora_ramanujan.sh
sbatch scripts/submit_qlora_ramanujan.sh

# Check job status
squeue -u $USER

# Watch live output  (replace JOBID with your actual number)
tail -f logs/slurm_lora_JOBID.out
```

---

## 📐 Evaluation Metrics

**Model Quality:** Validation Loss · Token-level Accuracy · F1 Score

**Efficiency:** Training Time · Peak GPU Memory · Trainable Parameters · Trainable %

**Inference:** Average Latency · P50 Latency · P95 Latency · Throughput · Peak GPU Memory

---

## 📊 Results

### Training and Model Quality

| Method | Train Time | Peak GPU Mem | Trainable % | Val Loss | Accuracy | F1 |
|---|---|---|---|---|---|---|
| Baseline | — | — | — | 1.6339 | 63.48% | 0.3666 |
| **LoRA** | **383 sec** | **7,598 MB** | **1.18%** | **1.0305** | **72.30%** | **0.3753** |
| QLoRA | 2,306 sec | 10,002 MB | ~1.18% | 1.0406 | 72.30% | 0.3739 |

LoRA reduced validation loss by **36.9%** relative to the zero-shot baseline while training fewer than 1.2% of total parameters. QLoRA achieved nearly identical accuracy at the cost of 6× longer training time, which comes from the per-step dequantisation overhead of the 4-bit base model.

### Inference Benchmarking

| Mode | Avg Latency | P50 Latency | P95 Latency | Throughput | Peak GPU Mem |
|---|---|---|---|---|---|
| Baseline | 6.02 s | 4.12 s | 4.99 s | 0.166 req/s | 2,959 MB |
| Fine-tuned | 6.05 s | 6.43 s | 6.45 s | 0.165 req/s | 3,085 MB |

The small latency increase in the fine-tuned mode is expected when LoRA adapters are not merged into the base model weights. Running `model.merge_and_unload()` before serving removes this overhead entirely.

---

## 📜 License

This project is released under the MIT License.