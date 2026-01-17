# Model Directory

This directory contains pretrained models used for the Monkey Verifier.

## Setup Instructions

Clone the LLaVA-RLHF model from Hugging Face:

```bash
cd ~/monkey-verifier/model_dir
git clone https://huggingface.co/zhiqings/LLaVA-RLHF-7b-v1.5-224
```

## Contents

| Model | Description | Source |
|-------|-------------|--------|
| `LLaVA-RLHF-7b-v1.5-224` | LLaVA model fine-tuned with RLHF, 7B parameters, 224px resolution | [Hugging Face](https://huggingface.co/zhiqings/LLaVA-RLHF-7b-v1.5-224) |

## Directory Structure

After setup, the directory should look like:

```
model_dir/
├── README.md
└── LLaVA-RLHF-7b-v1.5-224/
    ├── config.json
    ├── model weights...
    └── ...
```

## Notes

- Ensure you have `git-lfs` installed before cloning (`git lfs install`)
- The model download may take significant time depending on your connection speed
- Total download size is approximately 14GB

