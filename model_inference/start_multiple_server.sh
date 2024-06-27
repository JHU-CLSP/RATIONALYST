CURRENT_HOSTNAME=$(hostname)

CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1233 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
CUDA_VISIBLE_DEVICES=1 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1234 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
CUDA_VISIBLE_DEVICES=2 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1235 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
CUDA_VISIBLE_DEVICES=3 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1236 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
CUDA_VISIBLE_DEVICES=4 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1237 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME}&
CUDA_VISIBLE_DEVICES=5 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1238 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
CUDA_VISIBLE_DEVICES=6 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1239 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
CUDA_VISIBLE_DEVICES=7 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --port 1240 --max-logprobs 120000 --gpu-memory-utilization 0.5 > server_logs/${CURRENT_HOSTNAME} &
