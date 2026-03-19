# MedReason-Qwen

> Fine-tuned **Qwen 2.5-7B** on a medical reasoning dataset using SFT, LoRA, and Unsloth — on a free Kaggle T4 GPU.

---

## Overview

This project fine-tunes Qwen 2.5-7B, a 7-billion parameter LLM, to reason step-by-step through clinical diagnosis questions. The goal: teach an open-source model to think like a clinician — systematically, explicitly, and clearly — using a curated medical chain-of-thought dataset.

All training was done on a **free Kaggle T4 GPU** using aggressive memory optimizations.

---

## What It Does

- Loads the **ReasonMed** dataset — medical questions with detailed chain-of-thought reasoning
- Fine-tunes Qwen 2.5-7B using **Supervised Fine-Tuning (SFT)**
- Applies **LoRA adapters** to update only ~0.5% of model parameters
- Uses **Unsloth** for up to 2× faster training with significantly lower VRAM usage
- Runs end-to-end on a **16GB T4 GPU** (Kaggle free tier)

---

## Dataset

**ReasonMed** — available on HuggingFace

| Property | Detail |
|---|---|
| Type | Medical QA with chain-of-thought |
| Format | Question → Step-by-step reasoning → Answer |
| Samples used | 5,000 (subset) |

---

## Stack

### 1. Unsloth — Faster Fine-Tuning

Unsloth provides optimized kernels for transformer training, reducing VRAM usage significantly compared to vanilla Transformers + PEFT.

- 2× faster training speed
- Up to 70% less VRAM usage
- Drop-in replacement for HuggingFace `transformers`

### 2. PEFT — Parameter-Efficient Fine-Tuning

PEFT freezes the base model weights and trains only a small set of adapter layers. This makes fine-tuning feasible on consumer or free-tier GPUs.

### 3. LoRA — Low-Rank Adaptation

LoRA injects small trainable rank-decomposition matrices into the transformer's attention layers. Only ~0.5% of total parameters are updated, drastically reducing memory and compute requirements.

---

## Repository Structure

```
medreason-qwen/
├── notebooks/
│   └── train_qwen_medreason.ipynb   # Main training notebook (Kaggle-ready)                   # SFT trainer setup
├── configs/
│   └── lora_config.yaml             # LoRA hyperparameters
├── outputs/                         # Saved LoRA adapters (gitignored)
├── requirements.txt
└── README.md
```

---

## Quick Start

**1. Clone the repo**

```bash
git clone https://github.com/your-username/medreason-qwen.git
cd medreason-qwen
```

**2. Install dependencies**

> ⚠️ Use `uv` instead of `pip`. It resolves the dependency conflicts between `unsloth`, `triton`, and `bitsandbytes` that cause kernel crashes.

```bash
# Install uv
curl -Ls https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

**3. Run training**

Open `notebooks/train_qwen_medreason.ipynb` on Kaggle and run all cells, or run locally:

```bash
python src/train.py
```

---

## Configuration

**LoRA** (`configs/lora_config.yaml`):

```yaml
r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: [q_proj, k_proj, v_proj, o_proj]
bias: none
task_type: CAUSAL_LM
```

**Training arguments:**

```python
TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    max_steps                  = 100,
    learning_rate              = 2e-4,
    fp16                       = True,
    logging_steps              = 10,
    output_dir                 = "outputs",
)
```

---

## Lessons Learned

The first run crashed in 30 seconds. What followed: dependency conflicts, OOM errors, GPU crashes, and endless kernel restarts. Two small changes fixed everything.

| Problem | Fix |
|---|---|
| Kernel crash in <30s | Switch to `uv` — resolves `unsloth` + `triton` conflicts |
| OOM on T4 (16GB) | Limit training to 5K samples; reduce batch size to 2 |
| `bitsandbytes` not loading | Install via `uv`, not `pip` |
| Repeated kernel restarts | Stream or slice the dataset — never load it fully upfront |
| Slow training | Enable `use_flash_attention_2=True` in Unsloth model load |

---

## Results

**Training summary (Stage 1 — SFT)**

| Metric | Value |
|---|---|
| GPU | Kaggle T4 (16GB) |
| Trainable parameters | 40,370,176 / 7,655,986,688 (~0.53%) |
| Training rows | 5,000 |
| Total steps | 353 |
| Runtime | 182.0 min |
| Samples / sec | 0.26 |
| Steps / sec | 0.032 |
| Peak GPU memory | 11.9 GB |
| Final train loss | 0.7839 |

**Loss curve (step → loss)**

| Step | Training Loss |
|---|---|
| 25 | 1.3054 |
| 50 | 0.8286 |
| 75 | 0.7575 |
| 100 | 0.7493 |
| 125 | 0.7490 |
| 150 | 0.7494 |
| 175 | 0.7218 |
| 200 | 0.7606 |
| 225 | 0.7067 |
| 250 | 0.7427 |
| 275 | 0.7376 |
| 300 | 0.7122 |
| 325 | 0.7382 |
| 350 | 0.7212 |

Loss dropped sharply from **1.30 → 0.75** within the first 75 steps, then stabilised in the **0.71–0.76** range for the remainder of training — a healthy convergence pattern for SFT on a reasoning dataset.

## Disclaimer

This model is a **research experiment** and is not intended for clinical use. Medical decisions must always involve qualified healthcare professionals. This project is for educational and ML research purposes only.

---

## References

- [Unsloth](https://github.com/unslothai/unsloth)
- [PEFT by HuggingFace](https://github.com/huggingface/peft)
- [Qwen 2.5 on HuggingFace](https://huggingface.co/Qwen)
- [LoRA — Hu et al. 2021](https://arxiv.org/abs/2106.09685)

---

## License

MIT License. See `LICENSE` for details.
