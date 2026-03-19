# MedReason-Qwen
Fine-tuned Qwen 2.5-7B Model on a medical reasoning dataset using SFT, LoRA, and Unsloth — on a free Kaggle T4 GPU.

fine-tuned Qwen 2.5-7B, a 7-billion parameter large language model, to reason step-by-step through clinical diagnosis questions. The goal: teach a capable open-source LLM to think like a clinician — systematically, explicitly, and clearly — using a curated medical chain-of-thought dataset.
All training was done on a free Kaggle T4 GPU using aggressive memory optimizations.

🔬 What It Does

Loads the ReasonMed dataset — medical questions with detailed chain-of-thought reasoning
Fine-tunes Qwen 2.5-7B using Supervised Fine-Tuning (SFT)
Applies LoRA adapters to update only ~0.5% of model parameters
Uses Unsloth for up to 2× faster training with significantly lower VRAM usage
Runs end-to-end on a 16GB T4 GPU (Kaggle free tier)

Dataset — ReasonMed (HuggingFace) 
-Medical QA with chain-of-thought

1. Unsloth — Faster Fine-Tuning
Unsloth provides optimized kernels for transformer training, reducing VRAM usage significantly compared to vanilla Transformers + PEFT. Key benefits:

2× faster training speed
Up to 70% less VRAM usage
Drop-in replacement for HuggingFace transformers

2. PEFT — Parameter-Efficient Fine-Tuning
PEFT freezes the base model weights and trains only a small set of adapter layers. This makes fine-tuning feasible on consumer or free-tier GPUs.
3. LoRA — Low-Rank Adaptation
LoRA injects small trainable rank-decomposition matrices into the transformer's attention layers. Only ~0.5% of total parameters are updated, drastically reducing memory and compute requirements.
