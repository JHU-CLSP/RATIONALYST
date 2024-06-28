import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import json
import time
import numpy as np
import aiohttp
import asyncio
from tqdm import tqdm
from argparse import ArgumentParser


start_time = time.time()

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
    {
        "role": "user",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is 72",
    },
    {
        "role": "assistant",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n <BOT>Let's first calculate the amount of chips Natalia sold in May<EOT> Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. <BOT>We have already calculated the number of chips in May, now we should calculate the sum of chips in April and May<EOT> Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is 72",
    },
    {
        "role": "user",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. The answer is 10"
    },
    {
        "role": "assistant",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: <BOT>To find out how much Weng earns per minute, we need to divide her hourly rate by the number of minutes in an hour<EOT> Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. <BOT>Now, to find out her total earnings for 50 minutes, we multiply the rate per minute by the number of minutes she worked<EOT> Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. The answer is 10"
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. The answer is 5"
    },
    {
        "role": "assistant",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: <BOT>To determine how much money Betty initially has, we calculate half of the total wallet cost<EOT> In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. <BOT>Betty’s grandparents contribute twice what her parents gave, so we need to calculate this amount<EOT> Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. <BOT>To find out how much more Betty needs to save, we subtract the total contributions and her initial amount from the total cost of the wallet<EOT> This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. The answer is 5"
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39"
    },
    {
        "role": "assistant",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: <BOT>First, we need to calculate the total number of chocolates Leah and her sister had initially<EOT> Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. <BOT>After they ate 35 chocolates, we need to calculate how many chocolates they have left<EOT> After eating 35, they had 74 - 35 = 39. The answer is 39"
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33"
    },
    {
        "role": "assistant",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: <BOT>First, we need to calculate how many golf balls Michael had after losing 23 on tuesday<EOT> Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. <BOT>After losing 2 more on wednesday, we need to calculate how many golf balls he had left<EOT> After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240 When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again. The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160 To write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research. The answer is 400"
    },
    {
        "role": "assistant",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: <BOT>First, we need to calculate how many non-valuable files Brennan downloaded in the first round<EOT> The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. <BOT>Next, we need to calculate how many valuable files Brennan downloaded in the first round<EOT> The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240 <BOT>After downloading 400 new files, we need to calculate how many non-valuable files he downloaded<EOT> When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again. <BOT>Finally, we need to calculate how many valuable files he was left with<EOT> The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160 <BOT>Now we need to calculate the total number of valuable files he has left to write his research<EOT> To write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research. The answer is 400"
    },
]

ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['train'])

async def get_response(data, pbar: tqdm):
    previous = "Question: " + data['question'] + '\nAnswer:' + data['answer'].replace("#### ", "The answer is")
    
    new_messages = messages_start_rationale.copy()
    new_messages.append({
        "role": "user",
        "content": previous
    })
    url = 'http://c013:1236/v1/chat/completions'
    content = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": new_messages,
        "max_tokens": 1000,
        "temperature": 0.0,
        "stop_token_ids": [128001, 128009],
        "seed": 14
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
    
    response = agent_response['choices'][0]['message']['content']
    d = {"input": "Question: " + data['question'] + "\nAnswer: " + data['answer'].replace("#### ", "The answer is"), "output": response}
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


if __name__ == '__main__':
    start_time = time.time()
    
    write_file = open("llama3_output_with_score_new_without_final.txt", "w")
    chunks = [data_list[x:x+1000] for x in range(0, len(data_list), 1000)]
    for chunk in chunks:
        result = apply_async(chunk)
        for d in result:
            write_file.write(json.dumps(d) + '\n')

    write_file.close()
    print("Total time: ", time.time() - start_time)

