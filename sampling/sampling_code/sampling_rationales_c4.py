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
import random
import sys


start_time = time.time()

base_url = ['http://c014', 'http://c006', 'http://c002', 'http://c004']
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

selected = ['FreeLaw', 'StackExchange', 'Wikipedia (en)', 'DM Mathematics', 'Ubuntu IRC', 'HackerNews']
# selected = ['DM Mathematics']


messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationales should help you with predicting future text and should not contain the information of preceeding context. You can add rationales by enclosing it with <BOT> and <EOT>. Other than the rationales, please do not modify the original text."
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

messages_gsm8k = [
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33"
    },
    {
        "role": "assistant",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: <BOT>First, we need to calculate how many golf balls Michael had after losing 23 on tuesday<EOT> Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. <BOT>After losing 2 more on wednesday, we need to calculate how many golf balls he had left<EOT> After losing 2 more, he had 35 - 2 = 33 golf balls. <BOT>We are ready to output the final answer<EOT>The answer is 33"
    },
    {
        "role": "user",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240 When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again. The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160 To write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research. The answer is 400"
    },
    {
        "role": "assistant",
        "content": "Question: Brennan was researching his school project and had to download files from the internet to his computer to use for reference. After downloading 800 files, he deleted 70% of them because they were not helpful. He downloaded 400 more files but again realized that 3/5 of them were irrelevant. How many valuable files was he left with after deleting the unrelated files he downloaded in the second round?\nAnswer: <BOT>First, we need to calculate how many non-valuable files Brennan downloaded in the first round<EOT> The number of non-valuable files Brennan downloaded in the first round is 70/100*800 = <<70/100*800=560>>560 files. <BOT>Next, we need to calculate how many valuable files Brennan downloaded in the first round<EOT> The number of valuable files Brennan downloaded in the first round is 800-560 = <<800-560=240>>240 <BOT>After downloading 400 new files, we need to calculate how many non-valuable files he downloaded<EOT> When he downloaded 400 new files, there were 3/5*400= <<3/5*400=240>>240 non-useful files, which he deleted again. <BOT>Finally, we need to calculate how many valuable files he was left with<EOT> The total number of valuable files he downloaded in the second round is 400-240 = <<400-240=160>>160 <BOT>Now we need to calculate the total number of valuable files he has left to write his research<EOT> To write his research, Brennan had 160+240 = <<160+240=400>>400 useful files to reference to write his research. <BOT>We are ready to output the final answer<EOT>The answer is 400"
    },
]

messages_apps = [
    {
        "role": "user", 
        "content": "Question: \nComplete the method that takes a sequence of objects with two keys each: country or state, and capital. Keys may be symbols or strings. The method should return an array of sentences declaring the state or country and its capital. ## Examples ```python [{\'state\': \'Maine\', \'capital\': \'Augusta\'}] --> [\"The capital of Maine is Augusta\"] [{\'country\' : \'Spain\', \'capital\' : \'Madrid\'}] --> [\"The capital of Spain is Madrid\"] [{\"state\" : \'Maine\', \'capital\': \'Augusta\'}, {\'country\': \'Spain\', \"capital\" : \"Madrid\"}] --> [\"The capital of Maine is Augusta\", \"The capital of Spain is Madrid\"] ```\n\nAnswer:\ndef capital(capitals):\n    result = []\n    for s in capitals:\n        location = s.get(\'state\') or s.get(\'country\')\n        capital_city = s.get(\'capital\')\n        result.append(f\"The capital of {location} is {capital_city}\")\n    return result\n"
    },
    {
        "role": "assistant",
        "content": "Question: \nComplete the method that takes a sequence of objects with two keys each: country or state, and capital. Keys may be symbols or strings. The method should return an array of sentences declaring the state or country and its capital. ## Examples ```python [{\'state\': \'Maine\', \'capital\': \'Augusta\'}] --> [\"The capital of Maine is Augusta\"] [{\'country\' : \'Spain\', \'capital\' : \'Madrid\'}] --> [\"The capital of Spain is Madrid\"] [{\"state\" : \'Maine\', \'capital\': \'Augusta\'}, {\'country\': \'Spain\', \"capital\" : \"Madrid\"}] --> [\"The capital of Maine is Augusta\", \"The capital of Spain is Madrid\"] ```\n\nAnswer:\ndef capital(capitals):\n    <BOT>Initialize an empty list to store the result strings<EOT>\n    result = []\n    <BOT>Iterate over each dictionary in the input list<EOT>\n    for s in capitals:\n        <BOT>Try to get the value associated with the key \'state\'. If \'state\' is not present, use the value associated with the key \'country\'<EOT>\n        location = s.get(\'state\') or s.get(\'country\')\n        <BOT>Get the value associated with the key \'capital\'<EOT>\n        capital_city = s.get(\'capital\')\n        \n        <BOT>Create a formatted string with the location and capital city and append it to the result list<EOT>\n        result.append(f\"The capital of {location} is {capital_city}\")\n    \n    <BOT>Return the list of formatted strings<EOT>\n    return result\n\n"
    },
]

messages_the_pile = [
    {
        "role": "user",
        "content": "\n\nNot applicable.\n\nJZ takes responsibility for drafting the manuscript. JZ, DY and XS are the attending doctors of this patient. QS is responsible for Pathological results and WW is for Microbiological results. YS and XS is responsible for revision of the manuscript. All authors read and approved the final manuscript.\n\nAuthors' information {#FPar1}\n====================\n\nJiangnan Zhao and Dongmei Yuan are resident physicians-in-training, and Yi Shi, Qunli Shi, Weiping Wang and Xin Su are attending specialist physicians who dedicate their time to mentoring trainees"
    },
    {
        "role": "assistant",
        "content": "\n\nNot applicable.\n\nJZ takes responsibility for drafting the manuscript. <BOT>Indicates JZ's role, suggesting the next roles to be described will involve other tasks like attending doctors and result analysis<EOT> JZ, DY and XS are the attending doctors of this patient. QS is responsible for Pathological results and WW is for Microbiological results. <BOT>Following the pattern, specific tasks related to results analysis are now described, predicting further details on manuscript revision<EOT> YS and XS is responsible for revision of the manuscript. <BOT>The revision responsibility follows naturally after result analysis, likely leading to a final approval statement<EOT> All authors read and approved the final manuscript. <BOT>Final approval statement indicates closure of the roles and tasks, suggesting a summary or author information follows<EOT> \n\nAuthors\' information \{#FPar1\} <BOT>Transition to detailed author information section, predicting descriptions of each author\'s background<EOT> \n====================\n\nJiangnan Zhao and Dongmei Yuan are resident physicians-in-training, <BOT>Starting with the roles of resident physicians-in-training, predicting subsequent details about attending specialist physicians<EOT> and Yi Shi, Qunli Shi, Weiping Wang and Xin Su are attending specialist physicians who dedicate their time to mentoring trainees"
    },
    {
        "role": "user",
        "content": "\n */\nstatic int __ram_vrfy_inp(DB * dbp, VRFY_DBINFO * vdp, PAGE * h, db_pgno_t pgno, db_indx_t * nentriesp, uint32 flags)\n{\n\tENV * env;\n\tRINTERNAL * ri;\n\tVRFY_CHILDINFO child;\n\tVRFY_PAGEINFO * pip;\n\tint ret, t_ret, isbad;\n\tuint32 himark, i, offset, nentries;\n\tdb_indx_t * inp;\n\tuint8 * pagelayout, * p;\n\n\tenv = dbp->env;\n\tisbad = 0;\n\tmemzero(&child, sizeof(VRFY_CHILDINFO));\n\tnentries = 0;\n\tpagelayout = NULL;\n\tif((ret = __db_vrfy_getpageinfo(vdp, pgno, &pip)) != 0)\n\t\treturn ret;\n\tif(TYPE(h) != P_IRECNO) {\n\t\tret = __db_unknown_path(env, \"__ram_vrfy_inp\");\n\t\tgoto err;\n\t}\n\thimark = dbp->pgsize;\n\tif((ret = __os_malloc(env, dbp->pgsize, &pagelayout)) != 0)\n\t\tgoto err;\n\tmemzero(pagelayout, dbp->pgsize);\n\tinp = P_INP(dbp, h);\n\tfor(i = 0; i < NUM_ENT(h); i++) {\n\t\tif((uint8 *)inp+i >= (uint8 *)h+himark) {\n\t\t\tEPRINT((env, DB_STR_A(\"1046\", \"Page %lu: entries listing %lu overlaps data\", \"%lu %lu\"), (ulong)pgno, (ulong)i));\n\t\t\tret = DB_VERIFY_BAD;\n\t\t\tgoto err;\n\t\t}\n\t\toffset = inp[i];\n\t\t/*\n\t\t * Check that the item offset is reasonable:  it points\n\t\t * somewhere after the inp array and before the end of the\n\t\t * page"
    },
    {
        "role": "assistant",
        "content": "\n */\nstatic int __ram_vrfy_inp(DB * dbp, VRFY_DBINFO * vdp, PAGE * h, db_pgno_t pgno, db_indx_t * nentriesp, uint32 flags)\n{\n\tENV * env;\n\tRINTERNAL * ri;\n\tVRFY_CHILDINFO child;\n\tVRFY_PAGEINFO * pip;\n\tint ret, t_ret, isbad;\n\tuint32 himark, i, offset, nentries;\n\tdb_indx_t * inp;\n\tuint8 * pagelayout, * p;\n\n\tenv = dbp->env;\n\tisbad = 0;\n\tmemzero(&child, sizeof(VRFY_CHILDINFO));\n\tnentries = 0;\n\tpagelayout = NULL;\n\tif((ret = __db_vrfy_getpageinfo(vdp, pgno, &pip)) != 0)\n\t\treturn ret;\n\t<BOT>Checks the return value of __db_vrfy_getpageinfo and returns if an error occurred<EOT>\n\tif(TYPE(h) != P_IRECNO) {\n\t\tret = __db_unknown_path(env, \"__ram_vrfy_inp\");\n\t\tgoto err;\n\t<BOT>If the page type is not P_IRECNO, handles the error by setting ret and jumping to err<EOT>\n\t}\n\thimark = dbp->pgsize;\n\tif((ret = __os_malloc(env, dbp->pgsize, &pagelayout)) != 0)\n\t\tgoto err;\n\t<BOT>Attempts to allocate memory for pagelayout and handles the error if the allocation fails<EOT>\n\tmemzero(pagelayout, dbp->pgsize);\n\tinp = P_INP(dbp, h);\n\tfor(i = 0; i < NUM_ENT(h); i++) {\n\t\tif((uint8 *)inp+i >= (uint8 *)h+himark) {\n\t\t\tEPRINT((env, DB_STR_A(\"1046\", \"Page %lu: entries listing %lu overlaps data\", \"%lu %lu\"), (ulong)pgno, (ulong)i));\n\t\t\tret = DB_VERIFY_BAD;\n\t\t\tgoto err;\n\t\t<BOT>Checks for overlapping entries in the inp array and handles the error if an overlap is detected<EOT>\n\t\t}\n\t\toffset = inp[i];\n\t\t/*\n\t\t * Check that the item offset is reasonable:  it points\n\t\t * somewhere after the inp array and before the end of the\n\t\t * page\n\t\t\n"
    },
    {
        "role": "user",
        "content": "Suppose -s*w = -26*w. Suppose -15 = -5*j - w*j. Let g(b) = b**3 - 3*b**2 - 2*b + 3. What is g(j)?\n-3\nLet o(w) = -9 - 31214*w + 31220*w - 3. Determine o(1).\n-6\nLet y = -11 + 15. Suppose -4*c + 30 = -y*r + 50, 0 = -5*c - r - 25. Let q(d) = d + 5. Give q(c).\n0\nLet r = 2899 - 2906. Let i(s) = 2*s**2 + 12*s - 3. Calculate i(r).\n11\nLet q be 6/(-105) + 4199/595. Let m(t) = t**3 - 9*t**2 + 5*t + 5. Give m(q).\n-58\nLet d = 440 - 92"
    },
    {
        "role": "assistant",
        "content": "Suppose -s*w = -26*w. Suppose -15 = -5*j - w*j. Let g(b) = b**3 - 3*b**2 - 2*b + 3. What is g(j)?\n<BOT>First solve the equations to find j, then substitute j into g(b) to calculate g(j)<EOT>\n-3\nLet o(w) = -9 - 31214*w + 31220*w - 3. Determine o(1).\n<BOT>Substitute w = 1 into the function o(w) and simplify to determine its value<EOT>\n-6\nLet y = -11 + 15. Suppose -4*c + 30 = -y*r + 50, 0 = -5*c - r - 25. Let q(d) = d + 5. Give q(c).\n<BOT>First calculate y, then solve the equations to find c, and finally substitute c into q(d) to calculate q(c)<EOT>\n0\nLet r = 2899 - 2906. Let i(s) = 2*s**2 + 12*s - 3. Calculate i(r).\n<BOT>First calculate r, then substitute r into the function i(s) to calculate i(r)<EOT>\n11\nLet q be 6/(-105) + 4199/595. Let m(t) = t**3 - 9*t**2 + 5*t + 5. Give m(q).\n<BOT>First calculate q, then substitute q into the function m(t) to determine m(q)<EOT>\n-58\nLet d = 440 - 92\n<BOT>Calculate the value of d as 440 - 92<EOT>"
    }
]
# ds = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
# ds = datasets.load_dataset("monology/pile-uncopyrighted", split="train")

messages_start_rationale.extend(messages_the_pile)
messages_start_rationale.extend(messages_ecqa)
messages_start_rationale.extend(messages_gsm8k)
# messages_start_rationale.extend(messages_apps)

async def get_response(data, pbar: tqdm):    
    new_messages = messages_start_rationale.copy()
    conversations = tokenizer.apply_chat_template(
        new_messages,
        tokenize=True,
    )
    # print(len(conversations))
    new_messages.append({
        "role": "user",
        "content": data
    })
    conversations = tokenizer.apply_chat_template(
        new_messages,
        tokenize=True,
    )
    # print(len(conversations))
    url = get_url()
    # url = 'http://c008:1236/v1/chat/completions'
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
                # print("Error in calling remote agent server")
                return {"input": "", "output": ""}
    
    response = agent_response['choices'][0]['message']['content']
    d = {"input": data, "output": response}
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
    
    write_file = open("llama3_output_with_score_new_the_pile" + str(sys.argv[1]) + ".txt", "w")
    i, count = 0, 0
    temp_list = []
    temp_list_data = []
    count_total = {}
    for item in selected:
        count_total[item] = 0
    # split to four chunks
    lines = '\n'.join(open("filtered_pile_documents.bk.txt").readlines())
    ds = lines.split("---------------------------------------------")
    ds = ds[len(ds) // 4 * int(sys.argv[1]):len(ds) // 4 * (int(sys.argv[1]) + 1)]
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    # split lines with ""
    
    chunks = [ds[x:x+2000] for x in range(0, len(ds), 2000)]
    for chunk in chunks:
        result = apply_async(chunk)
        for d in result:
            write_file.write(json.dumps(d) + '\n')
    
    write_file.close()
    print("Total time: ", time.time() - start_time)
    print(i)

