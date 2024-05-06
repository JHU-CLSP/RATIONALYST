# python -m vllm.entrypoints.openai.api_server --model /weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/model_training/output --dtype float32  --api_key "token-abc123" --port 1234 --max-logprobs 1
python -m vllm.entrypoints.openai.api_server --model /weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/model_training/output --dtype bfloat16  --port 1235 --max-logprobs 1
