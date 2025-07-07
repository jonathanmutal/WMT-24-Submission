# TIM-UNIGE: Translation into Low-Resource Languages of Spain for WMT24

# TIM-UNIGE: Translation into Low-Resource Languages of Spain for WMT24

**Authors**: Jonathan Mutal, Lucía Ormaechea  
**Affiliation**: TIM, University of Geneva  
**GitHub**: [https://github.com/jonathanmutal/WMT-24-Submission](https://github.com/jonathanmutal/WMT-24-Submission)

## 📝 Overview

This repository contains the code and configurations used for the constrained submission to the **WMT 2024 Shared Task** on translating from Spanish into **two low-resource Iberian languages**:  
- **Aranese (spa-arn)**  
- **Aragonese (spa-arg)**  

Our approach combines:
- A **multistage fine-tuning** strategy  
- The use of **synthetic data** from LLMs (e.g., BLOOMZ) and **rule-based Apertium** translations  
- A multilingual backbone model (NLLB 600M)  

The best systems achieved:
- **spa-arn**: 30.1 BLEU  
- **spa-arg**: 61.9 BLEU  

---

## 📂 Repository Structure

```
WMT-24-Submission/
├── config/               # Training configs (HF/Accelerate)
├── data/                 # Scripts and files for data preparation
├── experiments/          # Scripts to run experiments
├── generation/           # Scripts for BLOOMZ synthetic data generation
├── scripts/              # Utility scripts (tokenization, preprocessing)
├── fine_tuning/          # Multistage fine-tuning procedures
└── evaluation/           # Evaluation scripts (BLEU, ChrF, TER, significance)
```

---

## 🔧 Installation & Requirements

This repository uses Python ≥ 3.8 and the following main dependencies:
- `transformers`
- `datasets`
- `sacrebleu`
- `sentencepiece`
- `accelerate`
- `evaluate`

To install:
```bash
pip install -r requirements.txt
```

---

## 🗂️ Datasets Used

### Parallel Data
| Corpus         | spa-arn | spa-arg | spa-oci |
|----------------|---------|---------|---------|
| OPUS           | ✗       | 60k     | 1.1M    |
| FLORES+        | 997     | 997     | 997     |

### Monolingual Data
| Corpus         | spa     | arn     | arg     | oci     |
|----------------|---------|---------|---------|---------|
| OPUS/NLLB      | 19M     | ✗       | 213k    | 739k    |
| PILAR          | ✗       | 322k    | 84k     | ✗       |

---

## 🧪 Training Pipeline

### Multistage Fine-tuning

1. **Stage 1**  
   - Pre-training on large synthetic/crawled corpora (OPUS, PILAR, Apertium)
   - Filtered to max 100 tokens
   - Best model: NLLB

2. **Stage 2**  
   - Fine-tuning with smaller high-quality data (e.g. PILAR, BLOOMZ)

3. **Stage 3**  
   - Final fine-tuning using validation/test sets (FLORES+DEV)

---

## 🧠 Synthetic Data Generation

### BLOOMZ Generation (Aranese)
- Fine-tuned `bloomz-560m` on Aranese PILAR corpus
- Used causal LM objective with early stopping
- Sampled completions from truncated Spanish inputs to produce ~59k Aranese sentences

### Apertium Translation
- Forward and backtranslation
- Used for all languages where Apertium modules were available

---

## 📊 Results

### Spanish → Aranese (spa-arn)

| Model               | BLEU | ChrF | TER  |
|---------------------|------|------|------|
| Apertium            | 28.8 | 49.4 | 72.3 |
| NLLB (Stage 3.ii)   | 30.1 | 49.8 | 71.5 |

### Spanish → Aragonese (spa-arg)

| Model                  | BLEU | ChrF | TER  |
|------------------------|------|------|------|
| Apertium               | 61.1 | 79.3 | 27.2 |
| NLLB-Translation       | 61.9 | 79.5 | 26.8 |
| NLLB-Post-Edition      | 61.0 | 78.9 | 27.2 |

Significance was tested using **paired approximate randomization (10,000 trials)**.

---

## 📈 Evaluation Metrics

- **BLEU** – `sacrebleu` implementation  
- **ChrF** – Character F-score  
- **TER** – Translation Error Rate  
- All scripts located in `evaluation/`

---

## 📚 Citation

If you use this work, please cite the WMT paper:

```bibtex
@inproceedings{mutal2024timunige,
  title = "{TIM-UNIGE}: Translation into Low-Resource Languages of Spain for {WMT24}",
  author = {Mutal, Jonathan and Ormaechea, Lucía},
  booktitle = "Proceedings of the Ninth Conference on Machine Translation",
  year = {2024},
  pages = {862--870}
}
```

---

## 🧭 Future Work

- Analyze impact of real vs. synthetic data ratios  
- Incorporate external resources: dictionaries, orthographic standards  
- Improve post-editing model performance
