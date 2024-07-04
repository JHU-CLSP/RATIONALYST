from transformers import AutoTokenizer
import datasets
import json
import time
import numpy as np
import re
import random
import re
import random
from tqdm import tqdm
import asyncio
import aiohttp
from argparse import ArgumentParser

base_url = ['http://c014']
base_completion_url = ['http://c002']
world_url = ['http://c006'] # 70B data world model
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_world_model_url():
    return random.choice(world_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

def get_world_model_url_completion():
    return random.choice(base_completion_url) + ':' + str(random.choice(ports)) + '/v1/completions'

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

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to generate future text given preceeding text. If you find text enclosed in <BOT> and <EOT> tokens, those are rationales that may or may not help you with future generations"
    },
]

list_of_end_prompts = ['the answer is', 'The correct answer is', 'The answer is', 'The answer is indeed', 'the correct answer is indeed', 'The correct answer is indeed']

def setup_datalist(dataset_name):
    if dataset_name == "arc":
        ds = datasets.load_dataset("allenai/ai2_arc", 'ARC-Challenge')
        data_list = list(ds['test'])
        return data_list

def get_previous(dataset_name, data):
    if dataset_name == "arc":
        previous = "Question: " + data['question'] + '\nChoices: '
        for i in range(len(data['choices']['text'])):
            previous += chr(ord('A') + i) + " - " + data['choices']['text'][i] + " "
        previous = previous[:-1] + '\nAnswer:'
        return previous

def get_messages(dataset_name):
    if dataset_name == "arc":
        messages_arc = [
  {"role": "system", "content": "Let's think step by step and solve this question. Please analyze each answer choice before output the answer. Do not output the answer before you analyzed each choices. After you have analyzed all answer choices, you can finish the response with The answer is:"},
  {"role": "user", "content": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\Choices: A - dry palms B - wet palms C - palms covered with oil D - palms covered with lotion\nAnswer: "},
  {"role": "assistant", "content": "Dry palms would likely produce more friction and heat because there are no additional substances to reduce the friction."},
  {"role": "user", "content": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nChoices: A - dry palms B - wet palms C - palms covered with oil D - palms covered with lotion.\nAnswer: Dry palms would likely produce more friction and heat because there are no additional substances to reduce the friction."},
  {"role": "assistant", "content": "Wet palms would produce less heat because water acts as a lubricant, reducing friction."},
  {"role": "user", "content": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nChoices: A - dry palms B - wet palms C - palms covered with oil D - palms covered with lotion.\nAnswer: Dry palms would likely produce more friction and heat because there are no additional substances to reduce the friction.\nWet palms would produce less heat because water acts as a lubricant, reducing friction."},
  {"role": "assistant", "content": "Oil also acts as a lubricant and would reduce friction, leading to less heat production."},
  {"role": "user", "content": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nChoices: A - dry palms B - wet palms C - palms covered with oil D - palms covered with lotion.\nAnswer: Dry palms would likely produce more friction and heat because there are no additional substances to reduce the friction.\nWet palms would produce less heat because water acts as a lubricant, reducing friction.\nOil also acts as a lubricant and would reduce friction, leading to less heat production."},
  {"role": "assistant", "content": "Lotion, similar to oil, would reduce friction and therefore produce less heat when rubbing the hands."},
  {"role": "user", "content": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nChoices: A - dry palms B - wet palms C - palms covered with oil D - palms covered with lotion.\nAnswer: Dry palms would likely produce more friction and heat because there are no additional substances to reduce the friction.\nWet palms would produce less heat because water acts as a lubricant, reducing friction.\nOil also acts as a lubricant and would reduce friction, leading to less heat production.\nLotion, similar to oil, would reduce friction and therefore produce less heat when rubbing the hands."},
  {"role": "assistant", "content": "The answer is: A"},
  {"role": "user", "content": "Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nChoices: A - The refrigerator door is smooth. B - The refrigerator door contains iron. C - The refrigerator door is a good conductor. D - The refrigerator door has electric wires in it.\nAnswer: "},
  {"role": "assistant", "content": "Smoothness does not affect the magnetic properties of a surface, so this is unlikely to be the correct answer."},
  {"role": "user", "content": "Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nChoices: A - The refrigerator door is smooth. B - The refrigerator door contains iron. C - The refrigerator door is a good conductor. D - The refrigerator door has electric wires in it.\nAnswer: Smoothness does not affect the magnetic properties of a surface, so this is unlikely to be the correct answer."},
  {"role": "assistant", "content": "Magnets stick to iron and other ferromagnetic materials, making this a strong possibility."},
  {"role": "user", "content": "Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nChoices: A - The refrigerator door is smooth. B - The refrigerator door contains iron. C - The refrigerator door is a good conductor. D - The refrigerator door has electric wires in it.\nAnswer: Smoothness does not affect the magnetic properties of a surface, so this is unlikely to be the correct answer.\nMagnets stick to iron and other ferromagnetic materials, making this a strong possibility."},
  {"role": "assistant", "content": "Conductivity is not related to magnetism, so this is not likely the correct answer."},
  {"role": "user", "content": "Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nChoices: A - The refrigerator door is smooth. B - The refrigerator door contains iron. C - The refrigerator door is a good conductor. D - The refrigerator door has electric wires in it.\nAnswer: Smoothness does not affect the magnetic properties of a surface, so this is unlikely to be the correct answer.\nMagnets stick to iron and other ferromagnetic materials, making this a strong possibility.\nConductivity is not related to magnetism, so this is not likely the correct answer."},
  {"role": "assistant", "content": "The presence of electric wires does not influence the magnetic attraction of magnets to the door."},
  {"role": "user", "content": "Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nChoices: A - The refrigerator door is smooth. B - The refrigerator door contains iron. C - The refrigerator door is a good conductor. D - The refrigerator door has electric wires in it.\nAnswer: Smoothness does not affect the magnetic properties of a surface, so this is unlikely to be the correct answer.\nMagnets stick to iron and other ferromagnetic materials, making this a strong possibility.\nConductivity is not related to magnetism, so this is not likely the correct answer.\nThe presence of electric wires does not influence the magnetic attraction of magnets to the door."},
  {"role": "assistant", "content": "The answer is: B"},
  {"role": "user", "content": "Question: A fold observed in layers of sedimentary rock most likely resulted from the\nChoices: A - cooling of flowing magma. B - converging of crustal plates. C - deposition of river sediments. D - solution of carbonate minerals."},
  {"role": "assistant", "content": "Cooling of magma typically leads to the formation of igneous rocks, not folds in sedimentary layers, so this is unlikely."},
  {"role": "user", "content": "Question: A fold observed in layers of sedimentary rock most likely resulted from the\nChoices: A - cooling of flowing magma. B - converging of crustal plates. C - deposition of river sediments. D - solution of carbonate minerals.\nAnswer: Cooling of magma typically leads to the formation of igneous rocks, not folds in sedimentary layers, so this is unlikely."},
  {"role": "assistant", "content": "The convergence of crustal plates can exert pressure on sedimentary rock layers, causing them to fold."},
  {"role": "user", "content": "Question: A fold observed in layers of sedimentary rock most likely resulted from the\nChoices: A - cooling of flowing magma. B - converging of crustal plates. C - deposition of river sediments. D - solution of carbonate minerals.\nAnswer: Cooling of magma typically leads to the formation of igneous rocks, not folds in sedimentary layers, so this is unlikely.\nThe convergence of crustal plates can exert pressure on sedimentary rock layers, causing them to fold."},
  {"role": "assistant", "content": "Deposition typically forms flat layers rather than folds, so this is not likely the cause of folds in sedimentary rocks."},
  {"role": "user", "content": "Question: A fold observed in layers of sedimentary rock most likely resulted from the\nChoices: A - cooling of flowing magma. B - converging of crustal plates. C - deposition of river sediments. D - solution of carbonate minerals.\nAnswer: Cooling of magma typically leads to the formation of igneous rocks, not folds in sedimentary layers, so this is unlikely.\nThe convergence of crustal plates can exert pressure on sedimentary rock layers, causing them to fold.\nDeposition typically forms flat layers rather than folds, so this is not likely the cause of folds in sedimentary rocks."},
  {"role": "assistant", "content": "The solution of carbonate minerals affects the chemical composition and can lead to dissolution features but does not typically cause folding."},
  {"role": "user", "content": "Question: A fold observed in layers of sedimentary rock most likely resulted from the\nChoices: A - cooling of flowing magma. B - converging of crustal plates. C - deposition of river sediments. D - solution of carbonate minerals.\nAnswer: Cooling of magma typically leads to the formation of igneous rocks, not folds in sedimentary layers, so this is unlikely.\nThe convergence of crustal plates can exert pressure on sedimentary rock layers, causing them to fold.\nDeposition typically forms flat layers rather than folds, so this is not likely the cause of folds in sedimentary rocks.\nThe solution of carbonate minerals affects the chemical composition and can lead to dissolution features but does not typically cause folding."},
  {"role": "assistant", "content": "The answer is: B"},
  {"role": "user", "content": "Question: As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?\nChoices: A - 31 pounds and 14 kilograms B - 31 pounds and 84 kilograms C - 186 pounds and 14 kilograms D - 186 pounds and 84 kilograms\nAnswer: "},
  {"role": "assistant", "content": "31 pounds and 14 kilograms cannot be correct because the mass of the astronaut is given as 84 kilograms, which does not change regardless of location."},
  {"role": "user", "content": "Question: As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?\nChoices: A - 31 pounds and 14 kilograms B - 31 pounds and 84 kilograms C - 186 pounds and 14 kilograms D - 186 pounds and 84 kilograms\nAnswer: This cannot be correct because the mass of the astronaut is given as 84 kilograms, which does not change regardless of location."},
  {"role": "assistant", "content": "While the mass, 31 pounds and 84 kilograms is correct, the weight on the Moon (31 pounds) would not be the same on Earth because of the difference in gravity."},
  {"role": "user", "content": "Question: As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?\nChoices: A - 31 pounds and 14 kilograms B - 31 pounds and 84 kilograms C - 186 pounds and 14 kilograms D - 186 pounds and 84 kilograms\nAnswer: This cannot be correct because the mass of the astronaut is given as 84 kilograms, which does not change regardless of location.\nWhile the mass is correct, the weight on the Moon (31 pounds) would not be the same on Earth because of the difference in gravity."},
  {"role": "assistant", "content": "The weight 186 pounds seems plausible for Earth given the astronaut's mass, but the mass of 14 kilograms is incorrect as the astronaut's mass is 84 kilograms."},
  {"role": "user", "content": "Question: As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?\nChoices: A - 31 pounds and 14 kilograms B - 31 pounds and 84 kilograms C - 186 pounds and 14 kilograms D - 186 pounds and 84 kilograms\nAnswer: This cannot be correct because the mass of the astronaut is given as 84 kilograms, which does not change regardless of location.\nWhile the mass is correct, the weight on the Moon (31 pounds) would not be the same on Earth because of the difference in gravity.\nThe weight seems plausible for Earth given the astronaut's mass, but the mass of 14 kilograms is incorrect as the astronaut's mass is 84 kilograms."},
  {"role": "assistant", "content": "186 pounds and 84 kilograms is correct because the mass of the astronaut remains 84 kilograms, and the weight on Earth would be approximately 186 pounds, given the difference in gravity between the Earth and the Moon."},
  {"role": "user", "content": "Question: As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?\nChoices: A - 31 pounds and 14 kilograms B - 31 pounds and 84 kilograms C - 186 pounds and 14 kilograms D - 186 pounds and 84 kilograms\nAnswer: This cannot be correct because the mass of the astronaut is given as 84 kilograms, which does not change regardless of location.\nWhile the mass is correct, the weight on the Moon (31 pounds) would not be the same on Earth because of the difference in gravity.\nThe weight seems plausible for Earth given the astronaut's mass, but the mass of 14 kilograms is incorrect as the astronaut's mass is 84 kilograms.\nThis is correct because the mass of the astronaut remains 84 kilograms, and the weight on Earth would be approximately 186 pounds, given the difference in gravity between the Earth and the Moon."},
  {"role": "assistant", "content": "The answer is: D"},
]
        return messages_arc

def get_normalized_answer(dataset_name, data):
    if dataset_name == "arc":
        return data['answerKey']
    
def get_normalized_prediction(dataset_name, prediction):
    if dataset_name == "arc":
        normalized_prediction = prediction.strip().replace(": ", "").split(' ')[0]
        return normalized_prediction


np.random.seed(14)
debug = True

async def get_heuristic(heuristic, previous_list, world_model, answer=None):
    if heuristic == "random":
        return random.randint(0, len(previous_list) - 1), None
    elif heuristic == "world_model_perplexity":
        probs_list, token_list = [], []
        
        for temp_previous in previous_list:
            if debug:
                print("Previous: " + temp_previous)
            message = [
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
            token = ''.join([item["token"] for item in world_response['choices'][0]['logprobs']['content']])
            
            baseline_prob = 0
            probs_list.append(prob - baseline_prob)
            token_list.append(token)
        return np.argmax(probs_list), token_list[0]
    else:
        print("Heuristic not implemented!")
        exit()

    
async def get_post_prob(responses, previous_with_rationale):
    prob_list = []
    for response in responses:
        following_length = len(tokenizer.tokenize(response))
        message_with_rationale = messages_start_rationale + [
            {
                "role": "user",
                "content": previous_with_rationale
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
        url = get_world_model_url_completion()
        content = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
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
        # decay with 0.9
        perplexity_with_rationale_list = [item * 0.9 ** i for i, item in enumerate(perplexity_with_rationale_list)]
        prob_list.append(sum(perplexity_with_rationale_list) / len(perplexity_with_rationale_list))
        if debug:
            print("Response: " + response)
            print("Perplexity: " + str(prob_list[-1]))
    return responses[np.argmax(prob_list)]

async def get_response(data, pbar: tqdm, heuristic: str, agent_model: str, world_model: str, agent_url: str, dataset: str):
    previous = get_previous(dataset, data)
    prediction, response_list = "", [""]
    normalized_answer, normalized_prediction = "", ""
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
            previous_temp = previous + " <BOT>" + chosen_world_model_response + "<EOT>"
        # if debug:
        #     print("Chosen response: " + str(chosen_previous))
        
        new_messages = get_messages(dataset).copy()
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
       
        # if debug:
        #     print("Previous: " + previous.split("Answer:")[1].strip())
            # for response in response_list:
            #     print("Partial response: " + response)
        
        # the real selection happens here
        if heuristic == "world_model_perplexity":
            response = await get_post_prob(response_list, previous_temp)
            response_list = [response]
        
        # update result
        async with asyncio.Lock():
            has_found_answer = False
            
            for term in list_of_end_prompts:
                if term in previous:
                    try:
                        prediction = previous.split(term)[1].replace('\n', ' ').strip()
                        normalized_prediction = get_normalized_prediction(dataset, prediction)
                        if normalized_prediction == "":
                            pass
                        else:
                            has_found_answer = True
                            break
                    except:
                        print("Error")
                if has_found_answer:
                    break
        if has_found_answer:
            break
        if trail == 14:
            print("Stuck in loop")
            break
        if debug:
            print("-------------------")

    d = {
        "question": data['question'],
        "prediction": prediction,
        "normalized_answer": get_normalized_answer(dataset, data),
        "normalized_prediction": normalized_prediction,
        "full_response": previous,
    }
    pbar.update(1)
    return d

def apply_async(data_list, heuristic, agent_model, world_model, agent_url, dataset):
    pbar = tqdm(total=len(data_list))
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    tasks = [loop.create_task(get_response(data, pbar, heuristic, agent_model, world_model, agent_url, dataset)) for data in data_list]
    result = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return result

if __name__ == '__main__':
    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument("--heuristic", type=str, default="random", help="Heuristic to use for selecting the next response")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="dataset to test with")
    parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Agent model to use for generating responses")
    # parser.add_argument("--agent_model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct", help="Agent model to use for generating responses")
    parser.add_argument("--agent_url", type=str, default="c013:1236", help="Agent model url")
    parser.add_argument("--write_file", type=str, default="output_arc.jsonl", help="File to write the output to")
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
    dataset = args.dataset
    write_file = open(args.write_file, 'w')
    debug = args.debug
    
    data_list = setup_datalist(args.dataset)
    
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
            if data['question'] in new_data_list:
                temp_data_list.append(data)
        data_list = temp_data_list
        debug = True
    
    if debug:
        # only test on 10 samples
        for i in range(10):
            apply_async([data_list[i]], heuristic, agent_model, world_model, agent_url, dataset)
        exit()
    
    result = apply_async(data_list, heuristic, agent_model, world_model, agent_url, dataset)
    for d in result:
        write_file.write(json.dumps(d) + '\n')
    write_file.close()
    print("Total TIME: ", time.time() - start_time)
