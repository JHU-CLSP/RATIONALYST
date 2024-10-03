# 𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓

𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓 is a model pre-trained on implicit rationales extracted from web text and reasoning datasets to provide process supervision. 𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓 generalizes over reasoning tasks with little human intervention while beating much larger models!

## 𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓 Installation
We assume the use of slurm for model training and inference.

To install other requirements: `pip install -r requirements.txt`


## Running experiments
### Step 1: Prefilter on The Pile
Due to the size of The Pile, we implement a pre-filtering process to identify reasoning-rich documents by (1) computing the average semantic embedding of representative reasoning training sets using a paragraph embedding model, and (2) selecting documents from unlabelled datasets that exceed a cosine similarity threshold

```
cd sampling/prefilter_code
sh start_multiple_server.sh
```

This process will break The Pile into different chunks and run pre-filtering in parallel

### Step 2: Sample rationales from The Pile and the training set of GSM8K and ECQA
Implicit rationales are often embedded in unlabelled text, reflecting natural thought processes in daily communication. Our extraction process aims to make these rationales explicit. 

```
cd sampling/sampling_code
python sampling_rationales_all_datasets.py/sampling_rationales_c4.py
python calculate_perplexity_sampled_rationale_new_method.py
python filter_rationale.py
```

The rationale extraction process requires a vllm server to be running elsewhere. Please update the URL in the script to connect to the correct vllm server. The prompts for extracting rationales are already included in the script.

Rationales extracted from GSM8K can be found [here](https://huggingface.co/datasets/Dongwei/reasoning_world_model)

### Step 3: 𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓 model training
The goal of 𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓 training is to develop a model that can generate implicit rationales to guide stepwise problem-solving during inference time. We choose to fine-tune a LLaMa-3-8B model on extracted rationales from step 2.

```
python parse_sampled_rationale.py
sbatch sbatch_llama_finetune.sh
```

This training script is adapted from [stanford-alpaca](https://github.com/tatsu-lab/stanford_alpaca)

Trained RATIONALYST can be found [here](https://huggingface.co/Dongwei/Rationalyst_reasoning_datasets)

### Step 4: Model inference
How does 𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓 work during inference time? 

It's a stepwise process: 1. 𝐑𝐀𝐓𝐈𝐎𝐍𝐀𝐋𝐘𝐒𝐓 generates rationale based on the current trajectory. 2. Agent LLM proposes multiple next reasoning steps. 3. Implicit rationales help estimate which step is most probable. 4. The best step is chosen, and the process repeats.

```
cd model_inference/evaluation_scripts
The entrance of all task inference is inference.py
Test on different task: change dataset name
With RATIONALYST (using implicit supervision): change heuristic to world_model
Without RATIONALYST: change heuristic to random
```
