# reasoning_world_model

## Step 1: Sample rationales for GSM8K
`python inference_llama3.py`

## Step 2: Parse rationales and create training data for LLAMA3 fine-tuning
`python parse_sampled_rationale.py`

## Step 3: LLAMA3 model training
`sbatch sbatch_llama_finetune.sh`
