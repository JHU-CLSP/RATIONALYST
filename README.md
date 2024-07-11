# reasoning_world_model

## Step 1: Prefilter on The Pile
`cd sampling/prefilter_code`

`sh start_multiple_server.sh`

## Step 2: Sample rationales from The Pile and the training set of GSM8K and ECQA
`cd sampling/sampling_code`

`python sampling_rationales_all_datasets.py/sampling_rationales_c4.py`

`python calculate_perplexity_sampled_rationale_new_method.py`

`python filter_rationale.py`

## Step 3: Parse rationales and create training data for Rationalyst fine-tuning
`python parse_sampled_rationale.py`

## Step 4: Rationalyst model training
`sbatch sbatch_llama_finetune.sh`

## Step 5: Model inference
With world model: `sbatch sbatch_llama_inference_world_model_single.sh`
Without world model: `sbatch sbatch_llama_inference_single.sh`
