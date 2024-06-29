import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
from datasets import load_dataset
import json
import time
import numpy as np
import aiohttp
import asyncio
from tqdm import tqdm
from argparse import ArgumentParser
import random


start_time = time.time()

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationals should help you with predicting future text and not repeating previous generations. You can add rationals by writing <BOT>rational<EOT>. Each <BOT> should pair with one <EOT>. These rationales should represent the hidden rationales unstated in the original text that facilitate the reasoning. Other than the additional rationales, please do not modify the original text!"
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

messages_ecqa = [
    {
        "role": "user",
        "content": "Question: He came across a raw item but his pack was full, he had to abandon something if he was to what the item?\nChoices: A - join, B - acquire, C - engage, D - maintain, E - remit\nAnswer: Join is not the type of the activity associated with item. Acquiring requires space in a pack. To create space in a full pack, one has to abandon an item. Engage is not related to item. Maintain an item does not require one to make space for it. One cannot remit raw item. The answer is B - acquire."
    },
    {
        "role": "assistant",
        "content": "Question: He came across a raw item but his pack was full, he had to abandon something if he was to what the item?\nChoices: A - join, B - acquire, C - engage, D - maintain, E - remit\nAnswer: <BOT>Let's look at choice A to see whether it's related to the word item in the question<EOT> Join is not the type of the activity associated with item. <BOT>Let's look at choice B to understand its meaning<EOT> Acquiring requires space in a pack. <BOT>Let's see whether the meaning of choice B is aligns with the question<EOT> To create space in a full pack, one has to abandon an item. <BOT>Let's look at choice C to see whether it's related to the word item in the question<EOT> Engage is not related to item. <BOT>Let's look at choice D to check whether it's related to the fact that the backpack is full, as mentioned in the question<EOT> Maintain an item does not require one to make space for it. <BOT>Let's look at choice E to check whether it's related to the word raw item in the question<EOT> One cannot remit raw item. The answer is B - acquire."
    },
    {
        "role": "user",
        "content": "Question: Where is aberdeen in the US located?\nChoices: A - washington, B - europe, C - scotland, D - maryland, E - south dakota\nAnswer: Aberdeen is located in Washington State. Washington state is located in US. Europe is not located in US. Scotland is not located in US. Aberdeen is not located in Maryland. Aberdeen is not located in South Dakota. The answer is A - washington." 
    },
    {
        "role": "assistant",
        "content": "Question: Where is aberdeen in the US located?\nChoices: A - washington, B - europe, C - scotland, D - maryland, E - south dakota\nAnswer: <BOT>Let's look at choice A to see whether it fulfills the requirement in question that aberdeen needs to be in this place<EOT> Aberdeen is located in Washington State. <BOT>Let's look at choice A to see whether it fulfills the requirement in question that the place needs to be in the US<EOT> Washington state is located in US. <BOT>Let's look at choice B to see whether it fulfills the requirement in question that the place needs to be in the US<EOT> Europe is not located in US. <BOT>Let's look at choice C to see whether it's fulfills the requirement in question that the place needs to be in the US<EOT> Scotland is not located in US. <BOT>Let's look at choice D to see whether it fulfills the requirement in question that aberdeen needs to be in this place<EOT> Aberdeen is not located in Maryland. <BOT>Let's look at choice E to see whether it fulfills the requirement in question that aberdeen needs to be in this place<EOT> Aberdeen is not located in South Dakota."
    },
]

messages_math = [
    {
        "role": "user",
        "content": "Question: The real numbers $a$ and $b$ satisfy \[\begin{pmatrix} 2 \\ a \\ -7 \end{pmatrix} \times \begin{pmatrix} 5 \\ 4 \\ b \end{pmatrix} = \mathbf{0}.\]Enter the ordered pair $(a,b).$ Answer: In general, $\mathbf{v} \times \mathbf{w} = \mathbf{0}$ if and only if the vectors $\mathbf{v}$ and $\mathbf{w}$ are proportional.  Thus, the vectors $\begin{pmatrix} 2 \\ a \\ -7 \end{pmatrix}$ and $\begin{pmatrix} 5 \\ 4 \\ b \end{pmatrix}$ are proportional.  Thus, \[\frac{5}{2} = \frac{4}{a} = \frac{b}{-7}.\]Solving, we find $(a,b) = \boxed{\left( \frac{8}{5}, -\frac{35}{2} \right)}.$"
    },
    {
        "role": "assistant",
        "content": "Question: The real numbers $a$ and $b$ satisfy \[\begin{pmatrix} 2 \\ a \\ -7 \end{pmatrix} \times \begin{pmatrix} 5 \\ 4 \\ b \end{pmatrix} = \mathbf{0}.\]Enter the ordered pair $(a,b).$ Answer: <BOT>Let's first understand the condition for the cross product of two vectors to be zero<EOT>In general, $\mathbf{v} \times \mathbf{w} = \mathbf{0}$ if and only if the vectors $\mathbf{v}$ and $\mathbf{w}$ are proportional. <BOT>Let's understand the condition for two vectors to be proportional<EOT>Thus, the vectors $\begin{pmatrix} 2 \\ a \\ -7 \end{pmatrix}$ and $\begin{pmatrix} 5 \\ 4 \\ b \end{pmatrix}$ are proportional. <BOT>Let's write the proportionality condition for the two vectors<EOT>Thus, \[\frac{5}{2} = \frac{4}{a} = \frac{b}{-7}.\] <BOT>Let's solve the proportionality condition to find the values of a and b<EOT>Solving, we find $(a,b) = \boxed{\left( \frac{8}{5}, -\frac{35}{2} \right)}.$"
    }
]

messages_mmlu_pro = [
    {
        "role": "user",
        "content": "Question: Which word best summarizes Weber's explanation of the development of formally rational law?\nChoices: A - Socialism., B - Legitimacy., C - Authority., D - Democracy., E - Bureaucracy., F - Conflict., G - Capitalism., H - Charisma., I - Co-operation., J - Tradition.\nThought Process: A: Let's think step by step. We refer to Wikipedia articles on jurisprudence for help. Weber explained the development of formal rationality in laws as how the modern society moved from tradition to rationality, where people decide actions based less on how they were culturally done and more on expected utilities. How rational individuals optimize efficiency of accomplishing tasks for higher rewards is a core principle of Capitalism. The answer is (G)."
    },
    {
        "role": "assistant",
        "content": "Question: Which word best summarizes Weber's explanation of the development of formally rational law?\nChoices: A - Socialism., B - Legitimacy., C - Authority., D - Democracy., E - Bureaucracy., F - Conflict., G - Capitalism., H - Charisma., I - Co-operation., J - Tradition.\nA: Let's think step by step. <BOT>We need to refer to prominant source of jurisprudence<EOT>We refer to Wikipedia articles on jurisprudence for help. Weber explained the development of formal rationality in laws as how the modern society moved from tradition to rationality, where people decide actions based less on how they were culturally done and more on expected utilities. <BOT>Let's try to link the Weber's explanation with answer choices<EOT> How rational individuals optimize efficiency of accomplishing tasks for higher rewards is a core principle of Capitalism. The answer is (G)."
    }
]

messages_start_rationale.extend(messages_ecqa)
messages_start_rationale.extend(messages_math)
messages_start_rationale.extend(messages_mmlu_pro)

gsm8k_data_list = load_dataset("gsm8k", 'main')
gsm8k_data_list = list(gsm8k_data_list['train'])
gsm8k_documents = ["Question: " + gsm8k_data['question'] + "\nAnswer: " + gsm8k_data['answer'] for gsm8k_data in gsm8k_data_list]
gsm8k_documents = [(item, 'gsm8k') for item in gsm8k_documents]

ecqa_documents = []
ecqa_key_mapping = {}
ecqa_dataset = load_dataset("tau/commonsense_qa")['train']
for line in open("/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/sampling/sampling_code/ecqa.jsonl", "r"):
    d = json.loads(line)
    ecqa_key_mapping[d['id']] = [d['positives'], d['negatives']]
    
for ecqa_data in ecqa_dataset:    
    positives = ' '.join(ecqa_key_mapping[ecqa_data['id']][0])
    negatives = ecqa_key_mapping[ecqa_data['id']][1]
    positive_insert_id = ord(ecqa_data['answerKey']) - ord('A')
    # insert the negative choice at the corrspoding position in positives
    negatives = negatives[:positive_insert_id] + [positives] + negatives[positive_insert_id:]
    temp_str = "Question: " + ecqa_data['question'] + "\nChoices: " 
    for i in range(len(ecqa_data["choices"]["text"])):
        temp_str += chr(ord('A') + i) + ' - ' + ecqa_data["choices"]["text"][i] + " "
    temp_str = temp_str[:-1] + '\n'
    temp_str += "Answer: " + ' '.join(negatives)
    temp_str += ' The answer is ' + ecqa_data['answerKey']
    ecqa_documents.append(temp_str)
ecqa_documents = [(item, 'ecqa') for item in ecqa_documents]

arc_data_list = load_dataset("ai2_arc", 'ARC-Challenge')['train']
arc_documents = []
for arc_data in arc_data_list:
    temp_str = "Question: " + arc_data['question'] + "\nChoices: "
    for i in range(len(arc_data['choices']['text'])):
        temp_str += chr(ord('A') + i) + ' - ' + arc_data['choices']['text'][i] + " "
    temp_str = temp_str[:-1] + '\n'
    temp_str += "Answer: " + arc_data['answerKey']
    arc_documents.append(temp_str)
arc_documents = [(item, 'arc') for item in arc_documents]

proofwriter_data_list = load_dataset("tasksource/proofwriter")['train']
proofwriter_documents = ["Question: " + proofwriter_data['question'] + "\nTheory: " + proofwriter_data['theory'] for proofwriter_data in proofwriter_data_list]

math_data_list = load_dataset("lighteval/MATH", "all")["train"]
math_documents = ["Question: " + math_data['problem'] + "\nAnswer: " + math_data['solution'] for math_data in math_data_list]
math_documents = [(item, 'math') for item in math_documents]

hellaswag_data_list = load_dataset("Rowan/hellaswag")['train']
hella_documents = ["Context: " + hellaswag_data['ctx'] + "\Endings:\n" + "\n".join(hellaswag_data['endings']) for hellaswag_data in hellaswag_data_list]
hella_documents = [(item, 'hellaswag') for item in hella_documents]

mmlu_pro_data_list = load_dataset("TIGER-Lab/MMLU-Pro")['validation']
mmlu_pro_documents = []
for mmlu_pro_data in mmlu_pro_data_list:
    temp_str = "Question: " + mmlu_pro_data['question'] + "\nChoices: " 
    for i in range(len(mmlu_pro_data['options'])):
        temp_str += chr(ord('A') + i) + ' - ' + mmlu_pro_data['options'][i] + " "
    temp_str = temp_str[:-1] + '\n'
    temp_str += mmlu_pro_data["cot_content"]
    mmlu_pro_documents.append(temp_str)
mmlu_pro_documents = [(item, 'mmlu_pro') for item in mmlu_pro_documents]

data_list = gsm8k_documents + ecqa_documents + math_documents + mmlu_pro_documents

# base_url = ['http://c014', 'http://c002', 'http://c004', "http://c010"]
base_url = ['http://c014']
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'


async def get_response(data, pbar: tqdm):
    previous = data[0]    
    new_messages = messages_start_rationale.copy()
    new_messages.append({
        "role": "user",
        "content": previous
    })
    url = get_url()
    content = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": new_messages,
        "max_tokens": 3000,
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
            except Exception as e:
                print(e)
                print("Error in calling remote agent server")
                return {"input": data, "output": "Error in calling remote agent server", "source": data[1]}
    
    response = agent_response['choices'][0]['message']['content']
    d = {"input": data, "output": response, "source": data[1]}
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
    
    write_file = open("sample_results_all_datasets.txt", "w")
    chunks = [data_list[x:x+2000] for x in range(0, len(data_list), 2000)]
    print(len(chunks))
    for chunk in chunks:
        result = apply_async(chunk)
        for d in result:
            write_file.write(json.dumps(d) + '\n')

    write_file.close()
    print("Total time: ", time.time() - start_time)

