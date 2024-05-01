#!/bin/bash
#SBATCH --job-name=llama_3_test
#SBATCH --time=24:0:0
#SBATCH --partition=defq
#SBATCH --nodes=1
# number of tasks (processes) per node
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16

#### load and unload modules you may need
# module unload openmpi/intel
# module load mvapich2/gcc/64/2.0b

#### execute code and write output file to OUT-24log.
# time mpiexec ./code-mvapich.x > OUT-24log
echo "Finished with job $SLURM_JOBID"

#### mpiexec by default launches number of tasks requested

source ~/.bashrc
cd /weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model
conda activate quiet-star

torchrun --nproc_per_node=2 --master_port=1234 llama3_finetune_alpaca.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --data_path llm_training_data_train.jsonl \
    --eval_data_path llm_training_data_eval.jsonl \
    --bf16 True \
    --output_dir output \
    --num_train_epochs 20 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
