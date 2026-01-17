# Data Directory

This directory contains datasets used for training and evaluation in the Monkey Verifier project.

## Setup Instructions

### 1. Bridge Data V2

Clone the Bridge Data V2 dataset and extract the images:

```bash
cd ~/monkey-verifier/data_dir
git clone https://huggingface.co/datasets/robomonkey-vla/bridge_data_v2
unzip bridge_v2_images.zip
```

### 2. Action Preference Bridge

Clone the action preference dataset:

```bash
cd ~/monkey-verifier/data_dir
git clone https://huggingface.co/datasets/robomonkey-vla/action_preference_bridge
```

## Contents

| Dataset | Description | Source |
|---------|-------------|--------|
| `bridge_data_v2` | Bridge V2 robot manipulation dataset with images | [Hugging Face](https://huggingface.co/datasets/robomonkey-vla/bridge_data_v2) |
| `action_preference_bridge` | Action preference annotations for RLHF training | [Hugging Face](https://huggingface.co/datasets/robomonkey-vla/action_preference_bridge) |

## Directory Structure

After setup, the directory should look like:

```
data_dir/
├── README.md
├── bridge_data_v2/
│   ├── bridge_v2_images.zip
│   └── ...
├── bridge_v2_images/          # extracted images
│   └── ...
└── action_preference_bridge/
    └── ...
```

## Notes

- Ensure you have `git-lfs` installed before cloning (`git lfs install`)
- The `bridge_v2_images.zip` must be extracted after cloning
- These datasets are used for reward model training and RLHF fine-tuning

