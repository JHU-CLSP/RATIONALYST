# reasoning_world_model

## Step 1: Sample rationales for GSM8K
`python inference_llama3.py`

## Step 2: Parse rationales and create training data for LLAMA3 fine-tuning
`python parse_sampled_rationale.py`

## Step 3: LLAMA3 model training
`sbatch sbatch_llama_finetune.sh`


## Step 4: Model inference
With world model: `sbatch sbatch_llama_inference_world_model_single.sh`
Without world model: `sbatch_llama_inference_single.sh`
