import json
import random
import vllm
from vllm import LLM
from vllm import SamplingParams
import re
import random
import requests
import re
import random
from tqdm import tqdm
import asyncio
import aiohttp
import time
import numpy as np
import editdistance
import sys


correct, total = 0, 0
lines = []

filter_with_score = True

base_url = ['http://c014']
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/completions'

# file_name = "/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/sampling/llama3_output_with_score_new_the_pile.txt"
file_name = sys.argv[1]

success_dict = {}
unsucess_dict = {}

for line in open(file_name):
    # print(line)
    if not line.startswith("-----"):
        total += 1
        try:
            d = json.loads(line.strip())
        except:
            print("error in reading")
            continue
        d['input'] = d['input'][0]
        if d['source'] not in success_dict:
            success_dict[d['source']] = 0
        # if d['source'] != 'math':
        #     continue
        # split string at the position of <BOT>
        splits = d['output'].split(" <BOT>")
        real_output = splits[0]
        for split in splits[1:]:
            try:
                leftover = split.split("<EOT>")[1]
                real_output += leftover
            except:
                continue
        input = d['input'].replace("\n", " ")
        
        str1 = d['input']
        str2 = real_output
        # we only keep the generated output if without rationales it's the same as the input
        # if d['input'].replace("\n", " ").replace(" ", '') == real_output.replace("\n", " ").replace("  ", " ").replace(' ', ''):
        # if d['input'].replace("\n", " ").replace(" ", '') == real_output.replace("\n", " ").replace(" ", ''):
        # print(1 - float(editdistance.eval(str1, str2)) / len(str1))
        if len(str1) > 0 and 1 - float(editdistance.eval(str1, str2)) / len(str1) >= 0.9:
            correct += 1
            # find the pairs of preceeding text and rationale
            splits = d['output'].split("<BOT>")
            preceeding = ""
            for i in range(1, len(splits)):
                try:
                    # get the preceeding text
                    preceeding += splits[i-1].split("<EOT>")[1]
                except:
                    preceeding += splits[i-1]
                # get the rationale
                rationale = splits[i].split("<EOT>")[0]
                # sometimes we can't find end of thought tokens
                try:
                    following = splits[i].split("<EOT>")[1]
                except:
                    continue
                # print("proceeding: " + preceeding)
                # print("rationale: " + rationale)
                # print("following: " + following)
                # print("----------------------------------------\n")
                # write_file.write(json.dumps({"preceeding": preceeding, "rationale": rationale, "following": following}) + "\n")
                lines.append({"preceeding": preceeding, "rationale": rationale, "following": following, "left_step": len(splits) - i})
            success_dict[d['source']] += 1
        else:
            print(str1)
            print(str2)
            print(d['output'])
            print("\n\n")
            unsucess_dict[d['source']] = unsucess_dict.get(d['source'], 0) + 1
print(success_dict)
print(unsucess_dict)

print("First round exact match with input: ")
print(correct, total, correct / total)
# shuffle the lines
random.shuffle(lines)

if not filter_with_score:
    exit()

# filter with score
messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to generate future text given preceeding text. If you find text enclosed in <BOT> and <EOT> tokens, those are rationales that may or may not help you with future generations"
    },
]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

async def get_response(data, pbar: tqdm):
    message = messages_start_rationale + [
        {
            "role": "user",
            "content": data['preceeding']
        },
        {
            "role": "assistant",
            "content": data["following"]
        }
    ]
    # print(data)
    if data["preceeding"] == "" or data["following"] == "" or data["rationale"] == "":
        pbar.update(1)
        return {"preceeding": data["preceeding"], "rationale": data["rationale"], "following": data["following"], "is_correct": False, "perplexity_without_rationale": 0, "perplexity_with_rationale": 0, "perplexity_without_rationale_list": [], "perplexity_with_rationale_list": []}
    following_message = [{"role": "assistant", "content": data["following"]}]
    following_length = len(tokenizer.tokenize(data["following"]))
    url = get_url()
    # print(url)
    content = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        # "model": world_model,
        "prompt": tokenizer.apply_chat_template(message, tokenize=False),
        "max_tokens": 0,
        "temperature": 0.0,
        "logprobs": 1,
        "echo": True,
    }
    headers = {
        "Content-Type": "application/json"
    }
    session_timeout = aiohttp.ClientTimeout(total=60000,sock_connect=6000,sock_read=6000)

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async with session.post(url, headers=headers, json=content) as world_response_baseline:
            try:
                world_response_baseline.raise_for_status()
                world_response_baseline = await world_response_baseline.json()
            except Exception as e:
                print(e)
                print("Error in calling remote world model")
    # print(len(world_response_baseline['choices'][0]['logprobs']['token_logprobs']), following_length)
    perplexity_without_rationale_list = world_response_baseline['choices'][0]['logprobs']['token_logprobs'][-following_length - 1: -1]
    previous_tokens = world_response_baseline['choices'][0]['logprobs']['tokens'][-following_length - 1: -1]
    previous_tokens = ''.join([token.replace("<|start_header_id|>", "") for token in previous_tokens])
    # print(previous_tokens)

    perplexity_without_rationale = 0
    # print(perplexity_without_rationale_list)
    for i in range(len(perplexity_without_rationale_list)):
        perplexity_without_rationale += perplexity_without_rationale_list[i] * np.power(0.9, i)
    
    message_with_rationale = messages_start_rationale + [
        {
            "role": "user",
            "content": data['preceeding'] + ' <BOT>' + data['rationale'] + '<EOT>'
        },
        {
            "role": "assistant",
            "content": data["following"]
        }
    ]
    url = get_url()
    # print(url)
    content = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        # "model": world_model,
        "prompt": tokenizer.apply_chat_template(message_with_rationale, tokenize=False),
        "max_tokens": 0,
        "temperature": 0.0,
        "logprobs": 1,
        "echo": True,
    }
    headers = {
        "Content-Type": "application/json"
    }
    session_timeout = aiohttp.ClientTimeout(total=60000,sock_connect=6000,sock_read=6000)
    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async with session.post(url, headers=headers, json=content) as world_response:
            try:
                world_response.raise_for_status()
                world_response = await world_response.json()
            except Exception as e:
                print(e)
                print("Error in calling remote world model")
    perplexity_with_rationale_list = world_response['choices'][0]['logprobs']['token_logprobs'][-following_length - 1: -1]
    tokens = world_response['choices'][0]['logprobs']['tokens'][-following_length - 1: -1]
    tokens = ''.join([token.replace("<|start_header_id|>", "") for token in tokens])
    # print(tokens)
    perplexity_with_rationale = 0
    # print(data['following'])
    # if previous_tokens != tokens:
    #     exit()
    
    for i in range(len(perplexity_with_rationale_list)):
        perplexity_with_rationale += perplexity_with_rationale_list[i] * np.power(0.9, i)
    
    d = {"preceeding": data['preceeding'], "rationale": data["rationale"], "following": data["following"], "is_correct": True, "perplexity_without_rationale": perplexity_without_rationale, "perplexity_with_rationale": perplexity_with_rationale, "perplexity_without_rationale_list": perplexity_without_rationale_list, "perplexity_with_rationale_list": perplexity_with_rationale_list,"previous_tokens": previous_tokens, "tokens": tokens}
    
    if perplexity_with_rationale < perplexity_without_rationale:
        d['is_correct'] = False
    pbar.update(1)
    return d
    

def apply_async(data_list):
    pbar = tqdm(total=len(data_list))
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_response(data, pbar)) for data in data_list]
    result = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return result

start_time = time.time()

write_file = open("filtered_rationales_total.jsonl", "w")
write_file_train, write_file_dev = open("filtered_rationales_train.jsonl", 'w'), open("filtered_rationales_dev.jsonl", 'w')

chunks = [lines[i:i + 1000] for i in range(0, len(lines), 1000)]
# chunks = [lines[i:i + 1] for i in range(0, len(lines), 1)]

res = []
res_total = []

for chunk in chunks:
    result = apply_async(chunk)
    for d in result:
        res_total.append((d, d['perplexity_with_rationale'] - d['perplexity_without_rationale']))
        if d['is_correct']:
            res.append(d)

res_total = sorted(res_total, key=lambda x: x[1])

for d in res_total:
    write_file.write(json.dumps(d[0]) + '\n')
write_file.close()
random.shuffle(res)
for i in range(0, int(len(res) * 0.95)):
    write_file_train.write(json.dumps(res[i]) + '\n')
for i in range(int(len(res) * 0.95), len(res)):
    write_file_dev.write(json.dumps(res[i]) + '\n')

print("Total TIME: ", time.time() - start_time)
