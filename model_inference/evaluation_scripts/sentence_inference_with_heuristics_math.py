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

client = AsyncOpenAI(
    http_client=httpx.AsyncClient(
            limits=httpx.Limits(
            max_connections=1000,
            max_keepalive_connections=100
        )
    )
)

base_url = ['http://c014']
base_big_url = ['http://c010']
# world_url = ['http://c010']  # total data world model
world_url = ['http://c006']  # small gsm8k world model
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]
big_ports = [1233, 1237]


def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_world_model_url():
    return random.choice(world_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_big_url():
    return random.choice(base_big_url) + ':' + str(random.choice(big_ports)) + '/v1/chat/completions'


def extract_and_convert_number_real(text):
    text = str(text)
    # deal with 2 + 3 = 5
    if '=' in text:
        text = text.split('=')[1].strip()
    # remove end dot
    if text.endswith('.'):
        text = text[:-1]
    # deal with 13.00 and 13
    if '.' in text:
        text = text.split('.')[0]
    pattern = re.compile(r'(\-?[0-9\.,]+)')
    match = pattern.search(text)
    if match:
        # Remove commas from the number, if any, and convert to float
        number_str = match.group().replace(',', '')
        return number_str
    else:
        return text

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

messages_math = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math problems. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. Please don't repeat your previous generation while you're generating the sentence. If you think you're ready to output the answer, you can wrap your answer with \boxed{}"
    },
]

messages_start_world_model = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
]
messages_start_world_model = []
messages_start_world_model_reward = [
    {
        "role": "system",
        "content": "Your task is to assign rewards to the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. The more probable the reasoning trajectory is, the higher the reward should be. Please only output one reward. The reward should be an integer in the range of 0 to 3."
    },
]

ds = datasets.load_dataset("lighteval/MATH", 'all')
data_list = list(ds['test'])
# data_list = [item for item in data_list if item['level'] == "Level 1"]

train_data_list = list(ds['train'])
# train_data_list = [item for item in train_data_list if item['level'] == "Level 1"]
np.random.seed(14)
debug = True
use_top = False
rand_list_from_train = np.random.choice(train_data_list, 9, replace=False)
for data in rand_list_from_train:
    l = []
    d = {"role": "user", "content": "Question: " + data['problem'] + "\nAnswer:"}
    l.append(d)
    answers = data['solution'].split(". ")
    for answer in answers:
        if answer == "":
            continue
        l.append({"role": "assistant", "content": answer})
        temp_d = d.copy()
        temp_d['content'] += answer + ". "
        l.append(temp_d)
        d = temp_d.copy()
    l.pop()
    messages_math.extend(l)

async def get_heuristic(heuristic, previous_list, world_model, answer=None):
    if heuristic == "gpt4_world":
        probs_list = []
        for temp_previous in previous_list:
            message = messages_start_world_model_reward + [
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
        return argmax_prob, None
    elif heuristic == "random":
        return random.randint(0, len(previous_list) - 1), None
    elif heuristic == "llama3_world":
        probs_list = []
        for temp_previous in previous_list:
            message = messages_start_world_model_reward + [
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
                print("world model output: ", world_response['choices'][0]['message']['content'])
            prob = sum(world_response['choices'][0]['logprobs']['token_logprobs']) / len(world_response['choices'][0]['logprobs']['token_logprobs'])
            probs_list.append(prob)
        return np.argmax(probs_list), None
    elif heuristic == "world_model_content":
        probs_list = []
        world_response_list = []
        for temp_previous in previous_list:
            message = messages_start_world_model + [
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
                print("world model output: ", world_response['choices'][0]['message']['content'])
            prob = sum(world_response['choices'][0]['logprobs']['token_logprobs']) / len(world_response['choices'][0]['logprobs']['token_logprobs'])
            probs_list.append(prob)
            world_response_list.append(world_response['choices'][0]['message']['content'])
        return np.argmax(probs_list), world_response_list[int(np.argmax(probs_list))]
    else:
        print("Heuristic not implemented!")
        exit()
    
async def get_response(data, pbar: tqdm, heuristic: str, agent_model: str, world_model: str, agent_url: str):
    previous = "Question: " + data['problem'] + "\nAnswer:"
    prediction, response_list = "", [""]
    normalized_answer, normalized_prediction = "", ""
    answer = data['solution']
    for trail in range(15):
        previous_list = []
        # for first step, the content in response list is empty, so the previous is the question
        for response in response_list:
            if "####" in response:
                response = response.replace("####", "The answer is:")
            temp_previous = previous + ' ' + response
            previous_list.append(temp_previous)
        
        chosen_previous, chosen_world_model_response = await get_heuristic(heuristic, previous_list, world_model)
        previous = previous_list[chosen_previous]
        
        # special case for when the heuristic is ''world_model_content'', the rationales are used to help with the generation of next action
        if chosen_world_model_response is not None and chosen_world_model_response.strip() != "We are ready to output the final answer":
            previous += " " + chosen_world_model_response
        if debug:
            print("Chosen response: " + str(chosen_previous))
        
        new_messages = messages_math.copy()
        # new_messages = messages_gsm8k_with_content.copy()
        new_messages.append({
            "role": "user",
            "content": previous
        })
        url = get_url()
        content = {
            "model": agent_model,
            "messages": new_messages,
            "max_tokens": 1000,
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
            response_list.append(response)
        
        if debug:
            print("Previous: " + previous.split("Answer:")[1].strip())
            for response in response_list:
                print("Partial response: " + response)
        
        # update result
        async with asyncio.Lock():
            has_found_answer = False
            previous_list = [previous_list[chosen_previous]]
            
            for i in range(len(previous_list)):
                item = previous_list[i]
                if remove_boxed(last_boxed_only_string(item)) is not None:
                    try:
                        # prediction = item.split("The answer is:")[1].replace('\n', ' ').strip()[:-1]
                        prediction = remove_boxed(last_boxed_only_string(item))
                        normalized_prediction = prediction
                        normalized_answer = remove_boxed(last_boxed_only_string(answer))
                        # answer = answer.replace('\\boxed', 'boxed') + "."
                        # normalized_answer = '}'.join(answer.split('boxed{')[1].split('}')[0: -1])
                        if normalized_prediction == "":
                            print("Prediction is empty")
                        else:
                            # print("final_answer: " + str(prediction) + '\t' + str(answer))
                            # if normalized_prediction == normalized_answer:
                            has_found_answer = True
                            previous = item
                            break
                    except:
                        print("Error")
            if has_found_answer:
                break
            if trail == 14:
                print("Stuck in loop")
                break

        if trail == 14:
            print("Stuck in loop")
            break
    d = {
        "question": data['problem'],
        "answer": answer,
        "prediction": prediction,
        "normalized_answer": normalized_answer,
        "normalized_prediction": normalized_prediction,
        "full_response": previous.split("Answer:")[1].strip(),
    }
    pbar.update(1)
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
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")
    # parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--agent_url", type=str, default="c013:1236", help="Agent model url")
    parser.add_argument("--write_file", type=str, default="output_math.json", help="File to write the output to")
    parser.add_argument("--world_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="World model to use for generating responses")
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
    write_file = open(args.write_file, 'w')
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
            print("Question: " + data_list[i]['problem'])
            apply_async([data_list[i]], heuristic, agent_model, world_model, agent_url)
        exit()
    
    chunks = [data_list[x:x+1700] for x in range(0, len(data_list), 1700)]
    for chunk in chunks:
        result = apply_async(chunk, heuristic, agent_model, world_model, agent_url)
        for d in result:
            write_file.write(json.dumps(d) + '\n')
    write_file.close()
    print("Total TIME: ", time.time() - start_time)
