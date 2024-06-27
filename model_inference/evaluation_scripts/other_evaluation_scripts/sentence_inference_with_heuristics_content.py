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

base_url = ['http://c005']
base_big_url = ['http://c011', 'http://c010']
world_url = ['http://c008']
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]
big_ports = [1233, 1237]

def get_world_url():
    return random.choice(world_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

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

messages_gsm8k = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math word problems. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. If you saw rationales enclosed in <BOT> and <EOT>, those are the rationales that help you with next sentence prediction, you can choose to follow them or neglect them as they are not part of the answre generation. Please don't repeat your previous generation while you're generating the sentence. If you think you're ready to output the answer, you can finish the response with The answer is:"
    },
]

messages_gsm8k_with_content = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math word problems. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. If you saw rationales enclosed in <BOT> and <EOT>, those are the rationales that help you with next sentence prediction, you can choose to follow them or neglect them as they are not part of the answer generation. Please don't repeat your previous generation while you're generating the sentence. If you think you're ready to output the answer or sense that you're looping, you can just finish the response with \"The answer is:\". If you think the rationales enclosed in <BOT> and <EOT> already gave the final asnwer, please just finsh the response with \"The answer is:\""
    },
    {
        "role": "user",
        "content": "Question: One logger can cut down 6 trees per day. The forest is a rectangle measuring 4 miles by 6 miles, and each square mile has 600 trees. If there are 30 days in each month, how many months will it take 8 loggers to cut down all the trees? \nAnswer: <BOT>First, we need to calculate the total area of the forest<EOT>"
    },
    {
        "role": "assistant",
        "content": "First find the total area of the forest: 4 miles * 6 miles = <<4*6=24>>24 square miles."
    },
    {
        "role": "user",
        "content": "Question: One logger can cut down 6 trees per day. The forest is a rectangle measuring 4 miles by 6 miles, and each square mile has 600 trees. If there are 30 days in each month, how many months will it take 8 loggers to cut down all the trees? \nAnswer: First find the total area of the forest: 4 miles * 6 miles = <<4*6=24>>24 square miles. <BOT>Next, calculate the total number of trees in the forest<EOT>"
    },
    {
        "role": "assistant",
        "content": "Then find the total number of trees in that area: 600 trees/square mile * 24 square miles = <<600*24=14400>>14400 trees."
    },
    {
        "role": "user",
        "content": "Question: One logger can cut down 6 trees per day. The forest is a rectangle measuring 4 miles by 6 miles, and each square mile has 600 trees. If there are 30 days in each month, how many months will it take 8 loggers to cut down all the trees? \nAnswer: First find the total area of the forest: 4 miles * 6 miles = <<4*6=24>>24 square miles. Then find the total number of trees in that area: 600 trees/square mile * 24 square miles = <<600*24=14400>>14400 trees. <BOT>Now, determine how many trees 8 loggers can cut down in a day<EOT>"
    },
    {
        "role": "assistant",
        "content": "Then find the number of trees all the loggers can cut down in one day: 8 loggers * 6 trees/logger = <<8*6=48>>48 trees."
    },
    {
        "role": "user",
        "content": "Question: One logger can cut down 6 trees per day. The forest is a rectangle measuring 4 miles by 6 miles, and each square mile has 600 trees. If there are 30 days in each month, how many months will it take 8 loggers to cut down all the trees? \nAnswer: First find the total area of the forest: 4 miles * 6 miles = <<4*6=24>>24 square miles. Then find the total number of trees in that area: 600 trees/square mile * 24 square miles = <<600*24=14400>>14400 trees. Then find the number of trees all the loggers can cut down in one day: 8 loggers * 6 trees/logger = <<8*6=48>>48 trees. <BOT>Next, divide the total number of trees by the number of trees cut down per day to find the total days needed<EOT>"
    },
    {
        "role": "assistant",
        "content": "Then divide the total number of trees by the number cut down per day to find the number of days it takes to clear the forest: 14400 trees / 48 trees/day = <<14400/48=300>>300 days."
    },
    {
        "role": "user",
        "content": "Question: One logger can cut down 6 trees per day. The forest is a rectangle measuring 4 miles by 6 miles, and each square mile has 600 trees. If there are 30 days in each month, how many months will it take 8 loggers to cut down all the trees? \nAnswer: First find the total area of the forest: 4 miles * 6 miles = <<4*6=24>>24 square miles. Then find the total number of trees in that area: 600 trees/square mile * 24 square miles = <<600*24=14400>>14400 trees. Then find the number of trees all the loggers can cut down in one day: 8 loggers * 6 trees/logger = <<8*6=48>>48 trees. Then divide the total number of trees by the number cut down per day to find the number of days it takes to clear the forest: 14400 trees / 48 trees/day = <<14400/48=300>>300 days. <BOT>Finally, convert the total days into months by dividing by the number of days in a month<EOT>"
    },
    {
        "role": "assistant",
        "content": "Then divide the time in days by the number of days per month to find the time in months: 300 days / 30 days/month = <<300/30=10>>10 months. The answer is: 10"
    },
    {
        "role": "user",
        "content": "Question: Andy is a lawyer who's working on two lawsuits. The first lawsuit has a 30% chance of paying out $5,000,000 upon a win and $0 if he loses it. The second lawsuit has a 50% chance of paying out $1,000,000 if Andy loses and a 50% chance of paying out $2,000,000 if he wins. Expressed as a percentage, how much more likely is it that Andy loses both lawsuits compared to winning both of them? \nAnswer: \n<BOT>First, we need to find the probability that Andy will lose the first lawsuit<EOT>"
    },
    {
        "role": "assistant",
        "content": "First find the probability that Andy will lose the first lawsuit: 100% - 30% = 70%."
    },
    {
        "role": "user",
        "content": "Question: Andy is a lawyer who's working on two lawsuits. The first lawsuit has a 30% chance of paying out $5,000,000 upon a win and $0 if he loses it. The second lawsuit has a 50% chance of paying out $1,000,000 if Andy loses and a 50% chance of paying out $2,000,000 if he wins. Expressed as a percentage, how much more likely is it that Andy loses both lawsuits compared to winning both of them? \nAnswer: First find the probability that Andy will lose the first lawsuit: 100% - 30% = 70%. <BOT>Next, multiply that probability by the probability he loses the second lawsuit<EOT>"
    },
    {
        "role": "assistant",
        "content": "Then multiply that probability by the probability he loses the second lawsuit: 70% * 50% = 35%."
    },
    {
        "role": "user",
        "content": "Question: Andy is a lawyer who's working on two lawsuits. The first lawsuit has a 30% chance of paying out $5,000,000 upon a win and $0 if he loses it. The second lawsuit has a 50% chance of paying out $1,000,000 if Andy loses and a 50% chance of paying out $2,000,000 if he wins. Expressed as a percentage, how much more likely is it that Andy loses both lawsuits compared to winning both of them? \nAnswer: First find the probability that Andy will lose the first lawsuit: 100% - 30% = 70%. Then multiply that probability by the probability he loses the second lawsuit: 70% * 50% = 35%. <BOT>Next, find the probability that Andy wins both lawsuits<EOT>"
    },
    {
        "role": "assistant",
        "content": "Then multiply the probabilities of Andy winning each lawsuit to find the probability he wins them both: 30% * 50% = 15%."
    },
    {
        "role": "user",
        "content": "Question: Andy is a lawyer who's working on two lawsuits. The first lawsuit has a 30% chance of paying out $5,000,000 upon a win and $0 if he loses it. The second lawsuit has a 50% chance of paying out $1,000,000 if Andy loses and a 50% chance of paying out $2,000,000 if he wins. Expressed as a percentage, how much more likely is it that Andy loses both lawsuits compared to winning both of them? \nAnswer: First find the probability that Andy will lose the first lawsuit: 100% - 30% = 70%. Then multiply that probability by the probability he loses the second lawsuit: 70% * 50% = 35%. Then multiply the probabilities of Andy winning each lawsuit to find the probability he wins them both: 30% * 50% = 15%. <BOT>Finally, subtract the probability of winning both from the probability of losing both to find the difference<EOT>"
    },
    {
        "role": "assistant",
        "content": "Then subtract the probability Andy wins both lawsuits from the probability he loses both lawsuits to find the difference: 35% - 15% = 20%. The answer is: 20"
    },
    {
        "role": "user",
        "content": "Question: Ruth prepared sandwiches. She ate 1 sandwich and gave 2 sandwiches to her brother. Her first cousin arrived and ate 2 sandwiches. Then her two other cousins arrived and ate 1 sandwich each. There were 3 sandwiches left. How many sandwiches did Ruth prepare? \nAnswer: <BOT>First, calculate the total sandwiches eaten by her two cousins<EOT>"
    },
    {
        "role": "assistant",
        "content": "Ruth's two cousins ate a total of 2 x 1 = <<2*1=2>>2 sandwiches."
    },
    {
        "role": "user",
        "content": "Question: Ruth prepared sandwiches. She ate 1 sandwich and gave 2 sandwiches to her brother. Her first cousin arrived and ate 2 sandwiches. Then her two other cousins arrived and ate 1 sandwich each. There were 3 sandwiches left. How many sandwiches did Ruth prepare? \nAnswer: Ruth's two cousins ate a total of 2 x 1 = <<2*1=2>>2 sandwiches. <BOT>Next, calculate the total sandwiches before her two cousins arrived<EOT>"
    },
    {
        "role": "assistant",
        "content": "So there were 3 + 2 = <<3+2=5>>5 sandwiches left before her two cousins arrived. The answer is: 5"
    },
    {
        "role": "user",
        "content": "Question: Ruth prepared sandwiches. She ate 1 sandwich and gave 2 sandwiches to her brother. Her first cousin arrived and ate 2 sandwiches. Then her two other cousins arrived and ate 1 sandwich each. There were 3 sandwiches left. How many sandwiches did Ruth prepare? \nAnswer: Ruth's two cousins ate a total of 2 x 1 = <<2*1=2>>2 sandwiches. So there were 3 + 2 = <<3+2=5>>5 sandwiches left before her two cousins arrived. <BOT>Now, calculate the total sandwiches before her first cousin arrived<EOT>"
    },
    {
        "role": "assistant",
        "content": "Before her first cousin arrived, there were 5 + 2 = <<5+2=7>>7 sandwiches."
    },
    {
        "role": "user",
        "content": "Question: Ruth prepared sandwiches. She ate 1 sandwich and gave 2 sandwiches to her brother. Her first cousin arrived and ate 2 sandwiches. Then her two other cousins arrived and ate 1 sandwich each. There were 3 sandwiches left. How many sandwiches did Ruth prepare? \nAnswer: Ruth's two cousins ate a total of 2 x 1 = <<2*1=2>>2 sandwiches. So there were 3 + 2 = <<3+2=5>>5 sandwiches left before her two cousins arrived. Before her first cousin arrived, there were 5 + 2 = <<5+2=7>>7 sandwiches. <BOT>Next, calculate the total sandwiches eaten by Ruth and her brother<EOT>"
    },
    {
        "role": "assistant",
        "content": "Ruth and her brother had 1 + 2 = <<1+2=3>>3 sandwiches altogether."
    },
    {
        "role": "user",
        "content": "Question: Ruth prepared sandwiches. She ate 1 sandwich and gave 2 sandwiches to her brother. Her first cousin arrived and ate 2 sandwiches. Then her two other cousins arrived and ate 1 sandwich each. There were 3 sandwiches left. How many sandwiches did Ruth prepare? \nAnswer: Ruth's two cousins ate a total of 2 x 1 = <<2*1=2>>2 sandwiches. So there were 3 + 2 = <<3+2=5>>5 sandwiches left before her two cousins arrived. Before her first cousin arrived, there were 5 + 2 = <<5+2=7>>7 sandwiches. Ruth and her brother had 1 + 2 = <<1+2=3>>3 sandwiches altogether. <BOT>Finally, calculate the total sandwiches Ruth prepared<EOT>"
    },
    {
        "role": "assistant",
        "content": "Therefore, Ruth prepared 7 + 3 = <<7+3=10>>10 sandwiches. The answer is: 10"
    },
    {
        "role": "user",
        "content": "Question: James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? \nAnswer: <BOT>First, calculate the total amount paid to the minor characters<EOT>"
    },
    {
        "role": "assistant",
        "content": "He pays the minor characters 15000 * 4 = $<<15000*4=60000>>60,000."
    },
    {
        "role": "user",
        "content": "Question: James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? \nAnswer: He pays the minor characters 15000 * 4 = $<<15000*4=60000>>60,000. <BOT>Next, calculate the total amount paid to each major character<EOT>"
    },
    {
        "role": "assistant",
        "content": "The major characters each get 15000 * 3 = $<<15000*3=45000>>45,000."
    },
    {
        "role": "user",
        "content": "Question: James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? \nAnswer: He pays the minor characters 15000 * 4 = $<<15000*4=60000>>60,000. The major characters each get 15000 * 3 = $<<15000*3=45000>>45,000. <BOT>Now, calculate the total amount paid to all the major characters<EOT>"
    },
    {
        "role": "assistant",
        "content": "So he pays the main characters 45000 * 5 = $<<45000*5=225000>>225,000."
    },
    {
        "role": "user",
        "content": "Question: James runs a TV show and there are 5 main characters and 4 minor characters. He pays the minor characters $15,000 each episode. He paid the major characters three times as much. How much does he pay per episode? \nAnswer: He pays the minor characters 15000 * 4 = $<<15000*4=60000>>60,000. The major characters each get 15000 * 3 = $<<15000*3=45000>>45,000. So he pays the main characters 45000 * 5 = $<<45000*5=225000>>225,000. <BOT>Finally, calculate the total amount paid to both the minor and major characters<EOT>"
    },
    {
        "role": "assistant",
        "content": "So in total he pays 225000 + 60000 = $<<225000+60000=285000>>285,000. The answer is: 285,000"
    },
    {
        "role": "user",
        "content": "Question: Nikola is saving up for a bag of ant food before he can start his ant farm. He wants 400 ants in his farm. Each ant needs 2 ounces of food. Every ounce of ant food costs $0.1. Nikola is saving up by raking leaves. He charges $5 to start a job and then he charges by the leaf. Each leaf he rakes costs 1 penny. He finally saved up enough money after he raked 6,000 leaves. How many jobs did he complete? \nAnswer: <BOT>First, calculate the total amount of food needed for the ants<EOT>"
    },
    {
        "role": "assistant",
        "content": "The ants will need 800 ounces of food because 400 x 2 = <<400*2=800>>800."
    },
    {
        "role": "user",
        "content": "Question: Nikola is saving up for a bag of ant food before he can start his ant farm. He wants 400 ants in his farm. Each ant needs 2 ounces of food. Every ounce of ant food costs $0.1. Nikola is saving up by raking leaves. He charges $5 to start a job and then he charges by the leaf. Each leaf he rakes costs 1 penny. He finally saved up enough money after he raked 6,000 leaves. How many jobs did he complete? \nAnswer: The ants will need 800 ounces of food because 400 x 2 = <<400*2=800>>800. <BOT>Next, calculate the total cost of the ant food<EOT>"
    },
    {
        "role": "assistant",
        "content": "The ant food will cost $80 because 800 x 0.1 = <<800*0.1=80>>80."
    },
    {
        "role": "user",
        "content": "Question: Nikola is saving up for a bag of ant food before he can start his ant farm. He wants 400 ants in his farm. Each ant needs 2 ounces of food. Every ounce of ant food costs $0.1. Nikola is saving up by raking leaves. He charges $5 to start a job and then he charges by the leaf. Each leaf he rakes costs 1 penny. He finally saved up enough money after he raked 6,000 leaves. How many jobs did he complete? \nAnswer: The ants will need 800 ounces of food because 400 x 2 = <<400*2=800>>800. The ant food will cost $80 because 800 x 0.1 = <<800*0.1=80>>80. <BOT>Now, calculate the amount earned from raking leaves<EOT>"
    },
    {
        "role": "assistant",
        "content": "He has earned $60 from the leaves he has raked because 6000 leaves * $0.01/leaf = $<<6000*0.01=60>>60."
    },
    {
        "role": "user",
        "content": "Question: Nikola is saving up for a bag of ant food before he can start his ant farm. He wants 400 ants in his farm. Each ant needs 2 ounces of food. Every ounce of ant food costs $0.1. Nikola is saving up by raking leaves. He charges $5 to start a job and then he charges by the leaf. Each leaf he rakes costs 1 penny. He finally saved up enough money after he raked 6,000 leaves. How many jobs did he complete?\n\nAnswer: The ants will need 800 ounces of food because 400 x 2 = <<400*2=800>>800. The ant food will cost $80 because 800 x 0.1 = <<800*0.1=80>>80. He has earned $60 from the leaves he has raked because 6000 leaves * $0.01/leaf = $<<6000*0.01=60>>60. <BOT>Next, calculate the remaining amount needed after earning from raking leaves<EOT>"
    },
    {
        "role": "assistant",
        "content": "He earned $20 from all his jobs because 80 - 60 = <<80-60=20>>20."
    },
    {
        "role": "user",
        "content": "Question: Nikola is saving up for a bag of ant food before he can start his ant farm. He wants 400 ants in his farm. Each ant needs 2 ounces of food. Every ounce of ant food costs $0.1. Nikola is saving up by raking leaves. He charges $5 to start a job and then he charges by the leaf. Each leaf he rakes costs 1 penny. He finally saved up enough money after he raked 6,000 leaves. How many jobs did he complete?\n\nAnswer: The ants will need 800 ounces of food because 400 x 2 = <<400*2=800>>800. The ant food will cost $80 because 800 x 0.1 = <<800*0.1=80>>80. He has earned $60 from the leaves he has raked because 6000 leaves * $0.01/leaf = $<<6000*0.01=60>>60. He earned $20 from all his jobs because 80 - 60 = <<80-60=20>>20. <BOT>Finally, calculate the number of jobs completed based on the remaining amount<EOT>"
    },
    {
        "role": "assistant",
        "content": "He completed 4 jobs because 20 / 5 = <<20/5=4>>4. The answer is: 4"
    },
    {
        "role": "user",
        "content": "Question: Maia is a freelance blogger working for different clients for whom she writes different articles every day. She gets 6 client requests every day and works on four of them each day. How many client requests will she have remaining to work on after 5 days? \nAnswer: <BOT>Let's first calculate the number of requests she has left each day<EOT>"
    },
    {
        "role": "assistant",
        "content": "If she receives 6 job requests every day and works on 4 every day, she remains with 6-4 = <<6-4=2>>2 requests."
    },
    {
        "role": "user",
        "content": "Question: Maia is a freelance blogger working for different clients for whom she writes different articles every day. She gets 6 client requests every day and works on four of them each day. How many client requests will she have remaining to work on after 5 days? \nAnswer: If she receives 6 job requests every day and works on 4 every day, she remains with 6-4 = <<6-4=2>>2 requests. <BOT>Next, we need to calculate the total number of remaining requests after 5 days<EOT>"
    },
    {
        "role": "assistant",
        "content": "After 5 days, she will be remaining with 5*2 = <<5*2=10>>10 requests which she hasn't worked on. The answer is: 10"
    },
    {
        "role": "user",
        "content": "Question: Ivan buys 48 apples to make 24 full-size pies. He ends up not using as many as he thought because he decides to make mini pies instead of full-size ones, which only use 1/2 an apple each. If Ivan bakes 24 mini pies, how many apples does he have leftover? \nAnswer: <BOT>Let's calculate the number of apples used for the mini pies<EOT>"
    },
    {
        "role": "assistant",
        "content": "Ivan wants to bake 24 pies which use 1/2 an apple each, 24 pies x 1/2 apple each = <<24*1/2=12>>12 apples."
    },
    {
        "role": "user",
        "content": "Question: Ivan buys 48 apples to make 24 full-size pies. He ends up not using as many as he thought because he decides to make mini pies instead of full-size ones, which only use 1/2 an apple each. If Ivan bakes 24 mini pies, how many apples does he have leftover?\n\nAnswer: Ivan wants to bake 24 pies which use 1/2 an apple each, 24 pies x 1/2 apple each = <<24*1/2=12>>12 apples. <BOT>Next, we need to subtract the used apples from the total to find the leftover apples<EOT>"
    },
    {
        "role": "assistant",
        "content": "Ivan uses 12 of his 48 apples, 48 - 12 = 36 apples. The answer is: 36"
    },
    {
        "role": "user",
        "content": "Question: Enrique has 2 132-page contracts that he needs to shred. His paper shredder will only allow him to shred 6 pages at a time. How many times will he shred 6 units of paper until all of the contracts are shredded?\nAnswer: <BOT>Let's first calculate the total number of pages to be shredded<EOT>"
    },
    {
        "role": "assistant",
        "content": "He has 2 contracts that are 132 pages each so that's 2*132 = <<2*132=264>>264 pages."
    },
    {
        "role": "user",
        "content": "Question: Enrique has 2 132-page contracts that he needs to shred. His paper shredder will only allow him to shred 6 pages at a time. How many times will he shred 6 units of paper until all of the contracts are shredded?\n\nAnswer: He has 2 contracts that are 132 pages each so that's 2*132 = <<2*132=264>>264 pages. <BOT>Next, we need to calculate the number of times he will need to shred 6 pages at a time<EOT>"
    },
    {
        "role": "assistant",
        "content": "He can only shred 6 pages at a time so it will take him 264/6 = <<264/6=44>>44 units. The answer is: 44"
    },
    {
        "role": "user",
        "content": "Question: If Stu has 9 books and Albert has 4 times as many books as Stu, how many books do Stu and Albert have in total? \nAnswer: <BOT>Let's calculate the number of books Albert has<EOT>"
    },
    {
        "role": "assistant",
        "content": "Albert has 4 * 9 books belonging to Stu = <<4*9=36>>36 books."
    },
    {
        "role": "user",
        "content": "Question: If Stu has 9 books and Albert has 4 times as many books as Stu, how many books do Stu and Albert have in total? \nAnswer: Albert has 4 * 9 books belonging to Stu = <<4*9=36>>36 books. <BOT>Now, we need to calculate the total number of books they have together<EOT>"
    },
    {
        "role": "assistant",
        "content": "So the pair have a combined total of 36 books belonging to Albert + 9 books belonging to Stu = <<36+9=45>>45 books. The answer is: 45"
    },
]

messages_start_world_model = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text. If you think you're ready to output the answer or sense that you're looping, you can just prompt the response with \"You can output using only \"The answer is:\"\"."
    },
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
use_top = True
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
        return 0, None
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
            message = messages_gsm8k_with_content + [
                {
                    "role": "user",
                    "content": temp_previous
                }
            ]
            url = 'http://c014:1235/v1/chat/completions'
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
            url = get_world_url()
            content = {
                "model": world_model,
                "messages": message,
                "max_tokens": 2000,
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
            world_response_list.append(world_response['choices'][0]['message']['content'])
        return np.argmax(probs_list), world_response_list[int(np.argmax(probs_list))]
    else:
        print("Heuristic not implemented!")
        exit()
    
async def get_response(data, pbar: tqdm, heuristic: str, agent_model: str, world_model: str, agent_url: str):
    previous = "Question: " + data['question'] + " \nAnswer:"
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
        # if chosen_world_model_response is not None and chosen_world_model_response.strip() != "We are ready to output the final answer":
        #     previous += " <BOT>" + chosen_world_model_response + "<EOT>"
        if debug:
            print("Chosen response: " + str(chosen_previous))
        
        new_messages = messages_gsm8k_with_content.copy()
        new_messages.append({
            "role": "user",
            "content": previous + " <BOT>" + chosen_world_model_response + "<EOT>"
        })
        url = get_big_url()
        content = {
            "model": agent_model,
            "messages": new_messages,
            "max_tokens": 1000,
            "temperature": 0.7,
            "stop_token_ids": [128001, 128009],
            "best_of": 3,
            "n": 3,
            "seed": 14,
            # "repetition_penalty": 1.2,
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
            print("Previous: " + previous)
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
        if debug:
            print("-------------------")

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
    parser.add_argument("--heuristic", type=str, default="world_model_content", help="Heuristic to use for selecting the next response")
    # parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Agent model to use for generating responses")
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
