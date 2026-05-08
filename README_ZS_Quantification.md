# ZS-Quantification

Lightweight quantization and KV Cache optimization framework for vehicle-side large language model deployment.

## Overview

This project focuses on efficient LLM inference under vehicle-edge-cloud collaborative environments.

### Core Features

- Weight Quantization
- KV Cache Quantization
- Hybrid Precision Inference
- Pareto-based Configuration Search
- Vehicle-Edge-Cloud Deployment

## Project Structure

```text
ZS-quantification/
├── configs/
├── core/
├── datasets/
├── experiments/
├── visualization/
├── deployment/
├── results/
├── requirements.txt
├── setup.py
├── run.py
└── README.md
```

## Environment Setup

```bash
conda create -n zs_quant python=3.10
conda activate zs_quant
pip install -r requirements.txt
```

## Example Commands

### INT4 Weight Quantization

```bash
python run.py --model qwen --weight_bits 4
```

### KV Cache Compression

```bash
python run.py --kv_bits 2
```

### Pareto Search

```bash
python run.py --search pareto
```

## Experimental Settings

Recommended Models:

- Qwen2-7B
- LLaMA2-7B
- Mistral-7B

Recommended Tasks:

- Long-context QA
- Vehicle Dialogue
- In-car Assistant

## Future Work

- Dynamic KV Routing
- Multi-RSU Inference Scheduling
- Vehicle-Cloud Collaborative Inference
- Multi-modal LLM Deployment

## License

MIT License
