<div align="center">

# 🌾 HindiKrishi

**Crop advisory AI for Indian farmers in Hindi**

[![Model](https://img.shields.io/badge/🤗%20Model-HuggingFace-yellow)](https://huggingface.co/me-nabi/farmer-advisory-hindi-qwen2.5-3b)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-21K%20examples-green)](https://huggingface.co/datasets/me-nabi/hindikrishi-farmer-advisory-dataset)
[![Demo](https://img.shields.io/badge/🚀%20Demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/me-nabi/farmer-advisory-demo)
[![Colab](https://img.shields.io/badge/Try%20on-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/drive/1QaMN2Jn75bOr9zYiTwzsiFMkPb6FzEMe?usp=sharing)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
</div>

## Overview

HindiKrishi is a domain-specific language model fine-tuned to provide Indian farmers with practical crop advisory in Hindi. It answers questions about crop diseases, pesticide recommendations, fertilizer management, government schemes, and farming best practices.

The model is fine-tuned from **Qwen 2.5 3B Instruct** using **QLoRA** (4-bit quantization + LoRA adapters) on a curated dataset of 21,069 instruction-response pairs.

## Key Features

- Answers in Hindi with specific pesticide names and dosages (ml/liter, kg/hectare)
- Covers 25+ major Indian crops
- Includes government scheme information (PMFBY, PM-KISAN, PMKSY)
- Every response includes KVK verification disclaimer
- Can run on consumer GPUs (6GB+ VRAM) using 4-bit quantization

## Example

**Hindi — Pesticide Recommendation**

Q: सरसों में माहू कीट के लिए कौन सा कीटनाशक छिड़कें?
A: सरसों में माहू कीट के लिए Imidacloprid का उपयोग करें। इसकी मात्रा
0.5-1 ग्राम प्रति लीटर पानी में मिलाकर छिड़कें। यह छिड़काव बुवाई
के 20-30 दिन बाद करें। अधिक जानकारी के लिए अपने नजदीकी KVK से
संपर्क करें।

**English — Fertilizer Advice**

Q: What fertilizer should I use for wheat?
A: A balanced NPK fertilizer is ideal. Apply 60 kg/ha urea (46% N),
20 kg/ha superphosphate, and 15 kg/ha muriate of potash.
Additionally, apply 50 kg/ha farmyard manure.


## Setup

```bash
git clone https://github.com/me-nabi/farmer-advisory-ai.git
cd farmer-advisory-ai
pip install -r requirements.txt
```

## Inference

```bash
python inference.py --question "गेहूं में पीला रतुआ रोग के लक्षण क्या हैं?"
```

Or in Python:

```python
from inference import load_model, ask

model, tokenizer = load_model()
print(ask(model, tokenizer, "धान में भूरा माहू कीट का नियंत्रण कैसे करें?"))
```

Requires a GPU with 6GB+ VRAM. Model loads in 4-bit quantization.

## Training

```bash
python train.py --dataset data/final_dataset_v4.jsonl --epochs 2 --batch_size 8
```

| Parameter | Value |
|---|---|
| Base Model | Qwen/Qwen2.5-3B-Instruct |
| Method | QLoRA (4-bit NF4 + LoRA r=16) |
| Training Data | 21,069 examples |
| Hardware | NVIDIA A100 80GB (RunPod) |
| Training Time | 83 minutes |
| Final Loss | 0.388 |
| Epochs | 2 |
| Batch Size | 32 |
| Learning Rate | 2e-4 |

## Dataset

21K instruction-response pairs from:

- **ICAR PDFs → GPT-4o-mini** — 4,335 Hindi Q&A from government advisory documents
- **Targeted generation** — 4,726 Hindi pairs covering 36 crops × 8 subtopics
- **KisanVaani** — English agriculture Q&A
- **Mahesh2841/Agriculture** — English agriculture Q&A
- **DigiGreen + CGIAR** — Hindi agriculture translations
- **Vikaspedia** — Hindi government advisory (scraped)
- **AI4Bharat Indic Anudesh** — Hindi instruction data

To generate more data:

```bash
python scripts/generate_data.py --api_key YOUR_OPENAI_KEY
```

## How Data Was Collected

1. **HuggingFace Datasets** — Downloaded existing agriculture datasets
2. **ICAR PDFs** — Extracted text from government publications, used GPT-4o-mini to generate Q&A pairs in Hindi
3. **Vikaspedia** — Scraped government agriculture portal using Playwright
4. **Synthetic Generation** — Systematically generated Q&A covering 25 crops × 8 subtopics using GPT-4o-mini
5. All data cleaned, deduplicated, and formatted into ChatML

## Project Structure

```
farmer-advisory-ai/
├── data/
│   └── final_dataset_v4.jsonl
├── scripts/
│   └── generate_data.py
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```
## Limitations

- Pesticide names and dosages should be verified with local KVK before use
- Government scheme details may not reflect latest updates
- Performs better on common crops (wheat, rice, cotton) than rare ones
- Not a substitute for professional agricultural advice

## Disclaimer

⚠️ For informational purposes only. Always confirm with your local KVK before applying any pesticide or fertilizer.

## Author

**Md Ehtasham Nabi** — AI Engineer

[HuggingFace](https://huggingface.co/me-nabi) 
