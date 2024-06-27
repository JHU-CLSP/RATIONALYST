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

messages_gsm8k = [
    {
        "role": "system",
        # "content": "You are a smart assistant that solves math word problems. If you think you're ready to output the final answer, you can finish the response with The answer is:"
        "content": "You are a smart math word problem solver. Given question, please give me the answer"
    },
    {
        "role": "user",
        "content": "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
    },
    {
        "role": "assistant",
        "content": "Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6"
    },
    {
        "role": "user",
        "content": "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
    },
    {
        "role": "assistant",
        "content": "Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5"
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
    },
    {
        "role": "assistant",
        "content": "Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39"
    },
    {
        "role": "user",
        "content": "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
    },
    {
        "role": "assistant",
        "content": "Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8"
    },
    {
        "role": "user",
        "content": "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
    },
    {
        "role": "assistant",
        "content": "Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9"
    },
    {
        "role": "user",
        "content": "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
    },
    {
        "role": "assistant",
        "content": "Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29"
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
    },
    {
        "role": "assistant",
        "content": "Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33"
    },
    {
        "role": "user",
        "content": "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
    },
    {
        "role": "assistant",
        "content": "Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8"
    }
]


ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['test'])


async def get_response(data, pbar: tqdm, heuristic: str, agent_model: str, world_model: str):
    previous = "Question: " + data['question']
    prediction = ""
    normalized_answer, normalized_prediction = "", ""
    answer = data['answer'].split("####")[1].strip()
    new_messages = messages_gsm8k.copy()
    new_messages.append({
        "role": "user",
        "content": previous
    })
    url = 'http://c008:1236/v1/chat/completions'
    content = {
        "model": agent_model,
        "messages": new_messages,
        "max_tokens": 1000,
        "temperature": 0.3,
        "stop_token_ids": [128001, 128009],
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
            except:
                print("Error in calling remote agent server")
    
    print("question: " + str(data['question']))
    response = agent_response['choices'][0]['message']['content']
    response = response.replace("#### ", "The answer is ")
    print("agent_response: " + str(response))
        
    try:
        prediction = response.split("The answer is")[1].strip()
        normalized_answer = extract_and_convert_number_real(answer)
        normalized_prediction = extract_and_convert_number_real(prediction)
        print("final_answer: " + str(prediction) + '\t' + str(answer))
    except:
        print("Error")
    
    d = {
        "question": data['question'],
        "answer": answer,
        "prediction": prediction,
        "normalized_answer": normalized_answer,
        "normalized_prediction": normalized_prediction,
        "full_response": response,
    }
    pbar.update(1)
    return d

def apply_async(data_list, heuristic, agent_model, world_model):
    pbar = tqdm(total=len(data_list))
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_response(data, pbar, heuristic, agent_model, world_model)) for data in data_list]
    result = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return result


if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--heuristic", type=str, default="random", help="Heuristic to use for selecting the next response")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--write_file", type=str, default="output.json", help="File to write the output to")
    parser.add_argument("--world_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="World model to use for generating responses")
    args = parser.parse_args()
    heuristic = args.heuristic 
    agent_model = args.agent_model
    world_model = args.world_model
    write_file = open(args.write_file, 'w')
    
    result = apply_async(data_list, heuristic, agent_model, world_model)
    # result = apply_async(list(ds['train'])[:1000], heuristic, agent_model, world_model)
    for d in result:
        write_file.write(json.dumps(d) + '\n')
    write_file.close()
    print("Total TIME: ", time.time() - start_time)
