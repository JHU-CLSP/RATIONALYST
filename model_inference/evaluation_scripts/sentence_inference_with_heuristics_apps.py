from tempfile import TemporaryFile
from exceptiongroup import catch
from sympy import false
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import json
import time
import numpy as np
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
import sys
from openai import AsyncOpenAI
import httpx
from argparse import ArgumentParser
from itertools import islice
import os
import csv

client = AsyncOpenAI(
    http_client=httpx.AsyncClient(
            limits=httpx.Limits(
            max_connections=1000,
            max_keepalive_connections=100
        )
    )
)

np.random.seed(14)
random.seed(14)
debug = True
trails = []


log_file = open('/weka/scratch/djiang21/nlp_proj/model/model_inference/debug_log.csv', 'a', newline='')
csv_writer = csv.writer(log_file)
csv_writer.writerow(['Question', 'Previous0', 'Previous1', 'Previous2', 'Previous3' 'Chosen Previous'])  # 写入表头

base_url = ['http://c014']
base_big_url = ['http://c010']
# world_url = ['http://c010']  # total data world model
world_url = ['http://c002']  # small gsm8k world model
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]
big_ports = [1233, 1237]

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_world_model_url():
    return random.choice(world_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'


messages_apps = [
    {
        "role": "system",
        "content": "You are a smart assistant that is doing code completion. All you generate should be code and not natural language sentences. IMPORTANT: Do not repeat any part of the previous partial answer. Also don't repeat the complete code to me when you are done. If you think you have completed the code, even partially, don't tell me the code is correct or generate natural language sentences, just finish the response with ```\nCode Complete!"
    },
]

messages_start_world_model = [
    {
        "role": "system",
        "content": "Your task is to assign rewards to the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. The more probable the reasoning trajectory is, the higher the reward should be. Please only output one reward. The reward should be an integer in the range of 0 to 3."
    },
]
messages_start_world_model = []

ds = datasets.load_dataset("codeparrot/apps", split="test")
start, end = 0, 700
introductory_data = []
for data in ds:
    if data['difficulty'] == 'introductory' and data['starter_code'] == "" and data['solutions'] != "":
        introductory_data.append(data)
data_list = introductory_data[start:end]


ds_train = datasets.load_dataset("codeparrot/apps", split="train")
introductory_data_train = []
for data in ds_train:
    if data['difficulty'] == 'introductory' and data['starter_code'] == "" and data['solutions'] != "":
        d = json.loads(data['solutions'].strip())
        # if len(d[0].split('\n')) < 30 and len(data['question']) < 2000:
        introductory_data_train.append(int(data['problem_id']))
        print(len(d[0].split('\n')))
train_data_list = introductory_data_train[start:end]

indices = []
indices = np.random.choice(train_data_list, 4, replace=False)
print(indices)

short_data = [ds_train[int(indices[i])] for i in range(len(indices))]
for data in short_data:
    l = []
    que = data['question']
    
    d = {"role": "user", "content": "Question: " + que + "\n```python\n"}
    formatted_d = d.copy()
    l.append(formatted_d)
    
    if '```\n\n' in data['solutions']:
        continue
    
    data['solutions'] = json.loads(data['solutions'])[0]
    answers = data['solutions'].strip().split("\n")
    answer_group = []
    temp_group = []
    for line in answers:
        if ('for' in line or 'if' in line or 'while' in line or 'def' in line) and '    ' not in line and ':' in line:
            answer_group.append(temp_group)
            temp_group = []
        temp_group.append(line)
    answer_group.append(temp_group)
    answers = []
    for temp in answer_group:
        answers.append('\n'.join(temp))
    print(answer_group)
    print(len(answer_group))
    
    for answer in answers:
        if answer == "":
            continue
        l.append({"role": "assistant", "content": answer})
        temp_d = d.copy()
        temp_d['content'] += "\n" + answer
        formatted_temp_d = temp_d.copy()
        # formatted_temp_d['content'] += '\n\n--- Please continue answering without repeating the partial answer above. ---\n'
        l.append(formatted_temp_d)
        d = temp_d.copy()
    l.pop()
    l[-1]['content'] += "\n```\nCode Complete!"
    messages_apps.extend(l)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

conversations = tokenizer.apply_chat_template(
    messages_apps,
    tokenize=True,
)
print(len(conversations))

async def get_heuristic(heuristic, previous_list, world_model, answer=None):
    if heuristic == "gpt4_world":
        probs_list = []
        for temp_previous in previous_list:
            message = messages_start_world_model + [
                {
                    "role": "user",
                    "content": temp_previous
                }
            ]
            world_response = await client.chat.completions.create(
                model='gpt-4-turbo',
                # model='gpt-3.5-turbo',
                messages=message,  # Ensure messages is a list
            )
            # print("gpt-4 output: ", world_response.choices[0].message.content)
            try:
                if "Reward:" in world_response.choices[0].message.content:
                    reward = int(world_response.choices[0].message.content.split("Reward:")[1].strip())
                else:
                    reward = int(world_response.choices[0].message.content.strip())
            except:
                reward = np.random.randint(0, 4)
            probs_list.append(reward)

        # debug mode
        # print("Previous list: ")
        # for item in previous_list:
        #     print(item)
        # print("World response probability: ")
        # print(probs_list)
        # print("Chosen response: " + previous_list[np.argmax(probs_list)])
        # print("Chosen world response: " + response_list[np.argmax(probs_list)])
        
        argmax_prob = np.argmax(probs_list)
        return argmax_prob, []
    elif heuristic == "random":
        return random.randint(0, len(previous_list) - 1), None
    elif heuristic == "llama3_world":
        probs_list = []
        for temp_previous in previous_list:
            message = messages_start_world_model + [
                {
                    "role": "user",
                    "content": temp_previous
                }
            ]
            url = get_url()
            content = {
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "messages": message,
                "max_tokens": 1000,
                "temperature": 0.0,
                "stop_token_ids": [128001, 128009],
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
                    except:
                        print("Error in calling remote world model")
                        break
            
            # print("gpt-4 output: ", world_response['choices'][0]['message']['content'])
            try:
                if "Reward:" in world_response['choices'][0]['message']['content']:
                    reward = int(world_response['choices'][0]['message']['content'].split("Reward:")[1].strip())
                else:
                    reward = int(world_response['choices'][0]['message']['content'].content.strip())
            except:
                reward = np.random.randint(0, 4)
            probs_list.append(reward)
        return np.argmax(probs_list), None
    elif heuristic == "world_model_perplexity":
        probs_list = []
        world_response_list = []
        for temp_previous in previous_list:
            message = messages_start_world_model + [
                {
                    "role": "user",
                    "content": temp_previous
                }
            ]
            url = get_world_model_url()
            content = {
                "model": world_model,
                "messages": message,
                "max_tokens": 1000,
                "temperature": 0.0,
                "stop_token_ids": [128001, 128009],
                "logprobs": True,
                "top_logprobs": 1,
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
                        break
            if debug:
                print("gpt-4 output: ", world_response['choices'][0]['message']['content'])
            prob = sum(world_response['choices'][0]['logprobs']['token_logprobs']) / len(world_response['choices'][0]['logprobs']['token_logprobs'])
            probs_list.append(prob)
            world_response_list.append(world_response['choices'][0]['message']['content'] + "\n\n" + str(prob))            
        return np.argmax(probs_list), world_response_list
    elif heuristic == "world_model_content":
        probs_list = []
        world_response_list = []
        for temp_previous in previous_list:
            message = [
                {
                    "role": "user",
                    "content": temp_previous
                }
            ]
            url = 'http://c003:1235/v1/chat/completions'
            content = {
                "model": world_model,
                "messages": message,
                "max_tokens": 1000,
                "temperature": 0.0,
                "stop_token_ids": [128001, 128009],
                "logprobs": True,
                "top_logprobs": 1,
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
                    except:
                        print("Error in calling remote world model")
                        break
            if debug:
                print("gpt-4 output: ", world_response['choices'][0]['message']['content'])
            prob = sum(world_response['choices'][0]['logprobs']['token_logprobs']) / len(world_response['choices'][0]['logprobs']['token_logprobs'])
            probs_list.append(prob)
            world_response_list.append(world_response['choices'][0]['message']['content'])
        return np.argmax(probs_list), world_response_list
    else:
        print("Heuristic not implemented!")
        exit()
    
async def get_response(data, pbar: tqdm, heuristic: str, agent_model: str, world_model: str, agent_url: str):
    global trails
    previous = "Question: " + data['question'] + "\n```python\n"

    prediction, response_list = "", [""]
    for trail in range(15):
        previous_list = []
        for response in response_list:
            temp_previous = previous + '\n' + response
            previous_list.append(temp_previous)
        
        chosen_previous, world_model_response = await get_heuristic(heuristic, previous_list ,world_model)

        previous = previous_list[chosen_previous]
        
        # special case for when the heuristic is ''world_model_content'', the rationales are used to help with the generation of next action
        if debug:
            if len(previous_list) == 4:
                csv_writer.writerow([previous_list[0].split("```python")[0],previous_list[0].split("```python")[1] + "\n\n\nRationale: \n" + world_model_response[0] + '\n\n', previous_list[1].split("```python")[1]+ "\n\n\nRationale: \n" + world_model_response[1], previous_list[2].split("```python")[1]+ "\n\n\nRationale: \n" + world_model_response[2],
                previous_list[3].split("```python")[1]+ "\n\n\nRationale: \n" + world_model_response[3],
                str(chosen_previous)])  # 写入 CSV 文件
            print("Chosen response: " + str(chosen_previous))
        
        new_messages = messages_apps.copy()
        new_messages.append({
            "role": "user",
            "content": previous
        })
        url = get_url()
        content = {
            "model": agent_model,
            "messages": new_messages,
            "max_tokens": 500,
            "temperature": 0.7,
            "stop_token_ids": [128001, 128009],
            "best_of": 3,
            "n": 3,
            "seed": 14,
        }
        headers = {
            "Content-Type": "application/json"
        }
        session_timeout = aiohttp.ClientTimeout(total=60000,sock_connect=6000,sock_read=6000)
        
        async with aiohttp.ClientSession(timeout=session_timeout) as session:
            async with session.post(url, headers=headers, json=content) as agent_response:
                try:
                    agent_response.raise_for_status()
                    agent_response = await agent_response.json()
                except Exception as e:
                    print(e)
                    print("Error in calling remote agent server")
                    break
        
        response_list = []
        
        for output in agent_response['choices']:
            response = output['message']['content']
            
            # answers = response.strip().split("\n")
            # answer_group = []
            # temp_group = []
            # temp_group.append(answers[0])
            # for line in answers[1:]:
            #     if ('for' in line or 'if' in line or 'while' in line or 'def' in line) and '    ' not in line and ':' in line:
            #         answer_group.append(temp_group)
            #         break
            #     else:
            #         temp_group.append(line)
            # answer_group.append(temp_group)
            
            # response = '\n'.join(answer_group[0])
            response_list.append(response)
        
        if debug:
            print("Previous: " + previous.strip())
            for response in response_list:
                print("Partial response: " + response)
        
        # update result
        async with asyncio.Lock():
            has_found_answer = False
            previous_list = [previous_list[chosen_previous]]
            
            for i in range(len(previous_list)):
                item = previous_list[i]
                if '```' in item.split("```python\n")[1] or 'Code Complete!' in item.split("```python\n")[1]:
                    try:
                        previous = item
                        has_found_answer = True
                        break
                    except:
                        print("Error")
            if has_found_answer:
                break
            if trail == 14:
                print("Stuck in loop")
                previous = item
                print(prediction)
                break

        if trail == 14:
            print("Stuck in loop")
            break
    if(previous.find("```python\n") == -1):
        print("Did not follow instruction")
        d = {
            "problem_id": data["problem_id"],
            "input_output": data["input_output"],
            "question": data['question'],
            "full_response": "",
            }
        trails.append(trail)
        return d
    if(previous.split("```python\n")[1].strip() == ''):
        print("No generation")
    full_response = ""
    if "Code Complete!" in previous:
        try:
            full_response = previous.split("```python\n")[1].split("Code Complete!")[0].strip().replace("```", "")
        except:
            print(full_response)
    elif "```" in previous.split("```python\n")[1].strip():
        full_response = previous.split("```python\n")[1].split("```")[0].strip()
    else:
        full_response = previous.split("```python\n")[1].strip()
    if "Here is the" in full_response:
        full_response = full_response.split("Here is the")[0]
    if full_response == "":
        print("Empty response")

    d = {
        "problem_id": data["problem_id"],
        "input_output": data["input_output"],
        "question": data['question'],
        "full_response": full_response,
        "full_response_previous": previous,
    }
    pbar.update(1)
    trails.append(trail)
    return d

def apply_async(data_list, heuristic, agent_model, world_model, agent_url):
    pbar = tqdm(total=len(data_list))
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_response(data, pbar, heuristic, agent_model, world_model, agent_url)) for data in data_list]
    result = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return result


if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    
    parser.add_argument("--heuristic", type=str, default="random", help="Heuristic to use for selecting the next response")
    # parser.add_argument("--heuristic", type=str, default="world_model_perplexity", help="Heuristic to use for selecting the next response")
    
    # parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")

    parser.add_argument("--agent_url", type=str, default="None", help="Agent model url")
    
    parser.add_argument("--write_file", type=str, default="output_apps.jsonl", help="File to write the output to")
    # parser.add_argument("--write_file", type=str, default="/weka/scratch/djiang21/nlp_proj/model/model_inference/output_dongwei_700.json", help="File to write the output to")
    # parser.add_argument("--write_file", type=str, default="/weka/scratch/djiang21/nlp_proj/model/model_inference/output_700_world_model.json", help="File to write the output to")
    # parser.add_argument("--write_file", type=str, default="/weka/scratch/djiang21/nlp_proj/model/model_inference/output_700_2x.json", help="File to write the output to")
    
    # parser.add_argument("--world_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="World model to use for generating responses")
    parser.add_argument("--world_model", type=str, default="/weka/scratch/djiang21/nlp_proj/model/model_training/output/checkpoint-1166", help="World model to use for generating responses")
    
    parser.add_argument("--debug", type=bool, default=False, help="Debug mode")
    
    parser.add_argument("--compare_file_good", type=str, default=None, help="file with higher accuracy")
    parser.add_argument("--compare_file_bad", type=str, default=None, help="file with lower accuracy")
    args = parser.parse_args()
    heuristic = args.heuristic 
    agent_model = args.agent_model
    world_model = args.world_model
    agent_url = args.agent_url
    compare_file_good = args.compare_file_good
    compare_file_bad = args.compare_file_bad
    debug = args.debug
    
    if compare_file_good is not None and compare_file_bad is not None:
        new_data_list = []
        lines_good, lines_bad = [], []
        lines_good = open(compare_file_good).readlines()
        lines_bad = open(compare_file_bad).readlines()
        for i in range(len(lines_good)):
            data_good = json.loads(lines_good[i].strip())
            data_bad = json.loads(lines_bad[i].strip())
            if data_good['normalized_prediction'] == data_good['normalized_answer'] and data_bad['normalized_prediction'] != data_bad['normalized_answer']:
                new_data_list.append(data_good['question'])
        temp_data_list = []
        for data in data_list:
            if data["question"] in new_data_list:
                temp_data_list.append(data)
        data_list = temp_data_list
        debug = True
    
    if debug:
        # only test on 10 samples
        for i in range(10):
            print("Question: " + data_list[i]['question'])
            apply_async([data_list[650 + i]], heuristic, agent_model, world_model, agent_url)
        exit()
        
    write_file = open(args.write_file, 'w')
    
    result = apply_async(data_list, heuristic, agent_model, world_model, agent_url)
    for d in result:
        write_file.write(json.dumps(d) + '\n')
    write_file.close()
    print("Total TIME: ", time.time() - start_time)
    print(trails)
    print(sum(trails) / len(trails))
