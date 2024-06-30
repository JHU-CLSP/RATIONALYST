CURRENT_HOSTNAME=$(hostname)

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --tensor-parallel-size 4 --port 1233 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --tensor-parallel-size 4 --port 1234 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
