Certainly! Below is the **English version in Markdown format** (`README.md`) for your project, following the same structure and requirements:

---

# JIT-Smart Experiment Reproduction and Evaluation Guide

This project is based on the [JIT-A/JIT-Smart](https://github.com/JIT-A/JIT-Smart) framework, designed to reproduce and evaluate the performance of large language models (e.g., GPT-3.5-Turbo) in just-in-time defect repair recommendation tasks. The experiments are organized into three research questions: RQ1, RQ2, and RQ3, covering dataset generation, model inference, and result evaluation.

---

## 🧰 Environment Requirements

This project requires **two separate Python environments**:

### 1. Small Model Environment (for JIT-Smart baseline)

- **Python Version**: `3.8`
- **Purpose**: Run the small model baseline as defined in the original JIT-Smart project

```bash
# Create environment using conda
conda create -n jit-small python=3.8
conda activate jit-small
pip install -r requirements_small.txt  # Adjust based on actual dependencies
```

> ⚠️ Note: The small model should be run strictly according to the setup in [JIT-A/JIT-Smart](https://github.com/JIT-A/JIT-Smart).

---

### 2. Large Model Environment (for GPT-based experiments)

- **Python Version**: `>=3.10`
- **Purpose**: Call LLM APIs (e.g., OpenAI) to generate repair suggestions and evaluate results

```bash

---

---

## 🚀 Large Model Experiment Pipeline

### Step 1: Generate Dataset

Run the following script to prepare input data for large model inference:

```bash
python generate_dataset.py
```

---

### Step 2: Run Large Model Inference (for RQ1 & RQ2)

Activate the large model environment and execute:

```bash
sh run.sh 5 gpt-3.5-turbo Full
```

✅ This command generates model outputs required for **RQ1 and RQ2**, saved in the `results/` folder.

---

### Step 3: Evaluate RQ1 and RQ2

Run the evaluation scripts:

```bash
sh eval.sh RQ1
sh eval.sh RQ2
```

---

### Step 4: Obtain RQ3 Results (Ablation Study)

RQ3 evaluates the impact of context configurations. Run the following commands **in sequence**:

```bash
sh run.sh 5 gpt-3.5-turbo Nfc   
sh run.sh 5 gpt-3.5-turbo One   
```

After both jobs complete, run:

```bash
sh eval.sh RQ3
```

This will compare performance across `Full`, `Nfc`, and `One` context settings, validating the effectiveness of contextual information.

---


## 📝 Notes

1. **API Cost Management**: The `run.sh` script makes multiple API calls — monitor usage carefully. Test with a small subset first.
2. **Network Stability**: API requests may fail due to connection issues. Retry logic is included, but manual monitoring is recommended.
3. **Environment Isolation**: Always activate the correct environment (`jit-small` or `jit-large`) before running scripts.
4. **Custom Models**: You can replace `gpt-3.5-turbo` with other models like `gpt-4` or `gpt-4-turbo` for ablation or comparison studies.

---

## 📚 Citation

If you use this codebase, please cite the original work:

```bibtex
@article{jit-smart,
  title={JIT-Smart: Just-in-Time Defect Prediction and Repair via Large Language Models},
  author={JIT-A Team},
  journal={IEEE Transactions on Software Engineering},
  year={2024}
}
```

---
