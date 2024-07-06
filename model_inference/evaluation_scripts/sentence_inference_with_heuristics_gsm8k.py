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


# base_url = ['http://c013', 'http://c007', 'http://c009']
# base_url = ['http://c014']
# ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]
# ports = [1233, 1234]
base_url = ['http://c014']
base_completion_url = ['http://c002']
# base_completion_url = ['http://c008']
base_big_url = ['http://c011', 'http://c010']
world_url = ['http://c006'] # 70B data world model
# world_url = ['http://c010'] # 8B data world model
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]
big_ports = [1233, 1237]


def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_world_model_url():
    return random.choice(world_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_world_model_url_completion():
    return random.choice(base_completion_url) + ':' + str(random.choice(ports)) + '/v1/completions'

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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct")
# llm = LLM(model='/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/model_training/runs/gsm8k_8B_world_model_70B_data/checkpoint-250/')

messages_gsm8k = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math word problems. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. Please don't repeat your previous generation while you're generating the sentence. If you think you're ready to output the answer, you can finish the response with The answer is:"
    },
]

messages_gsm8k_with_content_old = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
    {
        "role": "user",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n <BOT>Let's first calculate the amount of chips Natalia sold in May<EOT>"
    },
    {
        "role": "assistant",
        "content": "Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May."
    },
    {
        "role": "user",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n <BOT>Let's first calculate the amount of chips Natalia sold in May<EOT> Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. <BOT>We have already calculated the number of chips in May, now we should calculate the sum of chips in April and May<EOT>"
    },
    {
        "role": "assistant",
        "content": "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.The answer is 72"
    },
    {
        "role": "user",
        "content": "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n <BOT>To find out how much Weng earns per minute, we need to divide her hourly rate by the number of minutes in an hour<EOT>"
    },
    {
        "role": "assistant",
        "content": "Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute."
    },
    {
        "role": "user",
        "content": "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\n <BOT>To find out how much Weng earns per minute, we need to divide her hourly rate by the number of minutes in an hour<EOT> Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. <BOT>Now, to find out her total earnings for 50 minutes, we multiply the rate per minute by the number of minutes she worked<EOT>"
    },
    {
        "role": "assistant",
        "content": "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. The answer is 10"
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n <BOT>To determine how much money Betty initially has, we calculate half of the total wallet cost<EOT>"
    },
    {
        "role": "assistant",
        "content": "Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50."
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n <BOT>To determine how much money Betty initially has, we calculate half of the total wallet cost<EOT> Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. <BOT>Betty’s grandparents contribute twice what her parents gave, so we need to calculate this amount<EOT>"
    },
    {
        "role": "assistant",
        "content": "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30."
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\n <BOT>To determine how much money Betty initially has, we calculate half of the total wallet cost<EOT> Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. <BOT>Betty’s grandparents contribute twice what her parents gave, so we need to calculate this amount<EOT> Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. <BOT>To find out how much more Betty needs to save, we subtract the total contributions and her initial amount from the total cost of the wallet<EOT>"
    },
    {
        "role": "assistant",
        "content": "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. The answer is 5"
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n <BOT>First, we need to calculate the total number of chocolates Leah and her sister had initially<EOT>"
    },
    {
        "role": "assistant",
        "content": "Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74."
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\n <BOT>First, we need to calculate the total number of chocolates Leah and her sister had initially<EOT> Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. <BOT>After they ate 35 chocolates, we need to calculate how many chocolates they have left<EOT>"
    },
    {
        "role": "assistant",
        "content": "After eating 35, they had 74 - 35 = 39. The answer is 39"
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n <BOT>First, we need to calculate how many golf balls Michael had after losing 23 on tuesday<EOT>"
    },
    {
        "role": "assistant",
        "content": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35."
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n <BOT>First, we need to calculate how many golf balls Michael had after losing 23 on tuesday<EOT> Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. <BOT>After losing 2 more on wednesday, we need to calculate how many golf balls he had left<EOT>"
    },
    {
        "role": "assistant",
        "content": "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\n <BOT>First, we need to calculate how many non-valuable files Brennan downloaded in the first round<EOT>"
    },
    {
        "role": "assistant",
        "content": "Answer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files."
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\n <BOT>First, we need to calculate how many non-valuable files Brennan downloaded in the first round<EOT> Answer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. <BOT>Next, we need to calculate how many valuable files Brennan downloaded in the first round<EOT>"
    },
    {
        "role": "assistant",
        "content": "The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240."
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\n <BOT>First, we need to calculate how many non-valuable files Brennan downloaded in the first round<EOT> Answer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. <BOT>Next, we need to calculate how many valuable files Brennan downloaded in the first round<EOT> The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240. <BOT>After downloading 400 new files, we need to calculate how many non-valuable files he downloaded in the second round<EOT>"
    },
    {
        "role": "assistant",
        "content": "When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again."
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\n <BOT>First, we need to calculate how many non-valuable files Brennan downloaded in the first round<EOT> Answer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. <BOT>Next, we need to calculate how many valuable files Brennan downloaded in the first round<EOT> The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240. <BOT>After downloading 400 new files, we need to calculate how many non-valuable files he downloaded in the second round<EOT> When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again. <BOT>Finally, we need to calculate how many valuable files he was left with<EOT>"
    },
    {
        "role": "assistant",
        "content": "The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160."
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\n <BOT>First, we need to calculate how many non-valuable files Brennan downloaded in the first round<EOT> Answer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. <BOT>Next, we need to calculate how many valuable files Brennan downloaded in the first round<EOT> The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240. <BOT>After downloading 400 new files, we need to calculate how many non-valuable files he downloaded in the second round<EOT> When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again. <BOT>Finally, we need to calculate how many valuable files he was left with<EOT> The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160. <BOT>Now we need to calculate the total number of valuable files he has left to write his research<EOT>"
    },
    {
        "role": "assistant",
        "content": "To write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research. The answer is 400"
    },
]

messages_gsm8k_with_content = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math word problems. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. Please don't repeat your previous generation while you're generating the sentence. If you think you're ready to output the answer, you can finish the response with The answer is:"
    },
    {
        "role": "user",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Natalia sold 48/2 = <<48/2=24>>24 clips in May"
    },
    {
        "role": "user",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 = <<48/2=24>>24 clips in May."
    },
    {
        "role": "assistant",
        "content": "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. The answer is: 72"
    },
    {
        "role": "user",
        "content": "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute"
    },
    {
        "role": "user",
        "content": "Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute."
    },
    {
        "role": "assistant",
        "content": "Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. The answer is: 10"
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50"
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50."
    },
    {
        "role": "assistant",
        "content": "Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30"
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30."
    },
    {
        "role": "assistant",
        "content": "This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. The answer is: 5"
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74"
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74."
    },
    {
        "role": "assistant",
        "content": "After eating 35, they had 74 - 35 = 39. The answer is: 39"
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35"
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\n Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35."
    },
    {
        "role": "assistant",
        "content": "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is: 33"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files."
    },
    {
        "role": "assistant",
        "content": "The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240."
    },
    {
        "role": "assistant",
        "content": "When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240 When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again."
    },
    {
        "role": "assistant",
        "content": "The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240 When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again. The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160."
    },
    {
        "role": "assistant",
        "content": "To write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research. The answer is: 400"
    },
    {
        "role": "user",
        "content": "Question: A baker made 320 muffins. If she sold two-thirds of them in the morning and the rest in the afternoon, how many muffins did she sell in the afternoon?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "The baker sold two-thirds of the muffins in the morning, which amounts to 2/3 * 320 = <<2/3*320=213.33>>213 muffins"
    },
    {
        "role": "user",
        "content": "Question: A baker made 320 muffins. If she sold two-thirds of them in the morning and the rest in the afternoon, how many muffins did she sell in the afternoon?\nAnswer: The baker sold two-thirds of the muffins in the morning, which amounts to 2/3 * 320 = <<2/3*320=213.33>>213 muffins."
    },
    {
        "role": "assistant",
        "content": "In the afternoon, she sold 320 - 213 = <<320-213=107>>107 muffins. The answer is: 107"
    },
    {
        "role": "user",
        "content": "Question: Kevin read 15 pages of his book on Monday and three times as many on Tuesday. If his book has 120 pages, how many pages does he have left to read?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Kevin read 3 * 15 = <<3*15=45>>45 pages on Tuesday"
    },
    {
        "role": "user",
        "content": "Question: Kevin read 15 pages of his book on Monday and three times as many on Tuesday. If his book has 120 pages, how many pages does he have left to read? \nAnswer: Kevin read 3 * 15 = <<3*15=45>>45 pages on Tuesday."
    },
    {
        "role": "assistant",
        "content": "Kevin read a total of 15 + 45 = <<15+45=60>>60 pages in two days. He has 120 - 60 = <<120-60=60>>60 pages left to read. The answer is: 60"
    }
]

# messages_start_world_model = [
#     {
#         "role": "system",
#         "content": "Your task is to add rationales given to a piece of text. The rationals should help you with predicting future text and the next steps to solve the problem."
#     },
# ]

messages_start_world_model = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
]

messages_start_world_model = [
]

messages_start_world_model_reward = [
    {
        "role": "system",
        "content": "Your task is to assign rewards to the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. The more probable the reasoning trajectory is, the higher the reward should be. Please only output one reward. The reward should be an integer in the range of 0 to 3."
    },
]

ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['test'])

train_data_list = list(ds['train'])
np.random.seed(14)
debug = True
use_top = False
rand_list_from_train = np.random.choice(train_data_list, 9, replace=False)
for data in rand_list_from_train:
    l = []
    d = {"role": "user", "content": "Question: " + data['question'] + "\nAnswer:"}
    l.append(d)
    data['answer'] = data['answer'].replace("####", "The answer is:")
    answers = data['answer'].split("\n")
    for answer in answers:
        if answer == "":
            continue
        l.append({"role": "assistant", "content": answer})
        temp_d = d.copy()
        if not answer.endswith("."):
            answer = answer + "."
        temp_d['content'] += " " + answer
        l.append(temp_d)
        d = temp_d.copy()
    l.pop()
    messages_gsm8k.extend(l)

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
            prob = sum(item["logprob"] for item in world_response['choices'][0]['logprobs']['content'][:-1]) / (len(world_response['choices'][0]['logprobs']['content']) - 1)
            # print([item["token"] for item in world_response['choices'][0]['logprobs']['content']])
            
            baseline_prob = []
            temp_previous_baseline = ""
            
            '''
            # for i in range(len(world_response['choices'][0]['logprobs']["tokens"])):
            for i in range(1):
                message = messages_start_world_model + [
                    {
                        "role": "user",
                        "content": temp_previous
                    },
                    {
                        "role": "assistant",
                        "content": ''.join(world_response['choices'][0]['logprobs']["tokens"])
                    }
                ]
                url = get_world_model_url_completion()
                # url = get_world_model_url()
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
                            break
                baseline_prob = world_response_baseline['choices'][0]['logprobs']['token_logprobs'][-len(world_response['choices'][0]['logprobs']["tokens"]):]
                # print(world_response['choices'][0]['logprobs']["tokens"][i])
                # try:
                #     # if world_response['choices'][0]['logprobs']["tokens"][i] in world_response_baseline['choices'][0]['logprobs']['top_logprobs'][0]:
                #     #     baseline_prob.append(world_response_baseline['choices'][0]['logprobs']['top_logprobs'][0][world_response['choices'][0]['logprobs']["tokens"][i]])
                #     # else:
                #     #     baseline_prob.append(world_response_baseline['choices'][0]['logprobs']['top_logprobs'][0][world_response['choices'][0]['logprobs']["tokens"][i].strip()])
                #     baseline_prob.append(world_response_baseline['choices'][0]['logprobs']['top_logprobs'][0][world_response['choices'][0]['logprobs']["tokens"][i]])
                # except:
                #     baseline_prob.append(min(world_response_baseline['choices'][0]['logprobs']['top_logprobs'][0].values()))
                #     print("Error in getting baseline prob")
            
            # message = messages_start_world_model + [
            #     {
            #         "role": "user",
            #         "content": temp_previous
            #     },
            #     {
            #         "role": "assistant",
            #         "content": ''.join(world_response['choices'][0]['logprobs']["tokens"])
            #     }
            # ]
            # sampling_params = SamplingParams(temperature=0.0, max_tokens=2000, prompt_logprobs=1)
            # prompts = tokenizer.apply_chat_template(message, tokenize=False)
            
            # outputs = await run_lm_locally(sampling_params, prompts)

            # # Print the outputs.
            # for output in outputs:
            #     for i in range(len(output.prompt_logprobs) - 1, len(output.prompt_logprobs) - len(world_response['choices'][0]['logprobs']["tokens"]) - 1, -1):
            #         for item in output.prompt_logprobs[i]:
            #             item = output.prompt_logprobs[i][item]
            #             try:
            #                 if item.rank == 1:
            #                     baseline_prob.append(item.logprob)
            #             except Exception as e:
            #                 print(e)
            
            baseline_prob = sum(baseline_prob) / len(baseline_prob)
            if debug:
                print("baseline_prob: ", baseline_prob)
                print("prob: ", prob)
            '''
            baseline_prob = 0
            probs_list.append(prob - baseline_prob)
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
    previous = "Question: " + data['question'] + "\nAnswer:"
    prediction, response_list = "", [""]
    normalized_answer, normalized_prediction = "", ""
    answer = data['answer'].split("####")[1].strip()
    for trail in range(15):
        previous_list = []
        # for first step, the content in response list is empty, so the previous is the question
        for response in response_list:
            # if not response.endswith('.'):
            #     response = response + '.'
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
        
        new_messages = messages_gsm8k.copy()
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
            
        if use_top:
            content = {
                "model": agent_model,
                "messages": new_messages,
                "max_tokens": 1000,
                "temperature": 0.0,
                "stop_token_ids": [128001, 128009],
                "seed": 14,
            }
            headers = {
                "Content-Type": "application/json"
            }
            session_timeout = aiohttp.ClientTimeout(total=60000,sock_connect=6000,  sock_read=6000)
        
            async with aiohttp.ClientSession(timeout=session_timeout) as session:
                async with session.post(url, headers=headers, json=content) as agent_response:
                    try:
                        agent_response.raise_for_status()
                        agent_response = await agent_response.json()
                    except Exception as e:
                        print(e)
                        print("Error in calling remote agent server")
                        break
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
                if "The answer is:" in item:
                    try:
                        prediction = item.split("The answer is:")[1].replace('\n', ' ').strip()
                        normalized_prediction = extract_and_convert_number_real(prediction)
                        normalized_answer = extract_and_convert_number_real(answer)
                        if normalized_prediction == "":
                            # print("Prediction is empty")
                            pass
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
        "question": data['question'],
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
    parser.add_argument("--write_file", type=str, default="output.json", help="File to write the output to")
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
            print("Question: " + data_list[i]['question'])
            apply_async([data_list[i]], heuristic, agent_model, world_model, agent_url)
        exit()
    
    result = apply_async(data_list, heuristic, agent_model, world_model, agent_url)
    for d in result:
        write_file.write(json.dumps(d) + '\n')
    write_file.close()
    print("Total TIME: ", time.time() - start_time)
