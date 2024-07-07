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
import torch
import os

base_url = ['http://c014']
base_completion_url = ['http://c002']
world_url = ['http://c006'] # 70B data world model
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]
debug = True
num_choices = 3
seed = 14

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)


gsm8k_datalist = None
math_datalist = None
ecqa_datalist = None

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

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to generate future text given preceeding text. If you find text enclosed in <BOT> and <EOT> tokens, those are rationales that may or may not help you with future generations"
    },
]

def setup_datalist(dataset_name):
    if dataset_name == "arc":
        ds = datasets.load_dataset("allenai/ai2_arc", 'ARC-Challenge')
        data_list = list(ds['test'])
        return data_list
    elif dataset_name == "gsm8k":
        ds = datasets.load_dataset("gsm8k", 'main')
        data_list = list(ds['test'])
        global gsm8k_datalist
        gsm8k_datalist = list(ds['train'])
        return data_list
    elif dataset_name == "math":
        ds = datasets.load_dataset("lighteval/MATH", 'all')
        data_list = list(ds['test'])
        global math_datalist
        math_datalist = list(ds['train'])
        return data_list
    elif dataset_name == "ecqa":
        ds = datasets.load_dataset("tau/commonsense_qa")
        global ecqa_datalist
        ecqa_datalist = list(ds['train'])
        data_list = list(ds['validation'])
        return data_list
    elif dataset_name == "proofwriter":
        data_list = []
        d = json.loads(''.join(open("/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/model_inference/evaluation_scripts/proofwriter_test.json").readlines()))
        for item in d:
            data_list.append(item)
        return data_list
    elif dataset_name == "mmlu_pro":
        ds = datasets.load_dataset("TIGER-Lab/MMLU-Pro")
        data_list = list(ds['test'])
        return data_list

def get_previous(dataset_name, data):
    if dataset_name == "arc" or dataset_name == "ecqa":
        previous = "Question: " + data['question'] + '\nChoices: '
        for i in range(len(data['choices']['text'])):
            previous += chr(ord('A') + i) + " - " + data['choices']['text'][i] + " "
        previous = previous[:-1] + '\nAnswer:'
        return previous
    elif dataset_name == "gsm8k":
        return "Question: " + data['question'] + "\nAnswer:"
    elif dataset_name == "math":
        return "Question: " + data['problem'] + "\nAnswer:"
    elif dataset_name == "proofwriter":
        return "Question: Based on the context information, is the following statement true, false, or unknown? " + data['question'] + "\nContext: " + data['context'] + "\nChoices: A) True B) False C) Unknown\nAnswer: "
    elif dataset_name == "mmlu_pro":
        previous = "Question: " + data['question'] + "\nChoices: "
        for i in range(len(data['options'])):
            previous += chr(ord('A') + i) + " - " + data['options'][i] + " "
        previous = previous[:-1] + '\nAnswer:'
        return previous

def get_messages(dataset_name):
    np.random.seed(14)
    if dataset_name == "arc":
        messages_arc = [
  {"role": "system", "content": "Let's think step by step and solve this question. Please analyze each answer choice before output the answer. Do not output the answer before you analyzed each choices. After you have analyzed all answer choices, you can finish the response with The answer is:"},
   {"role": "user", "content": "Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?\nChoices: A - dry palms B - wet palms C - palms covered with oil D - palms covered with lotion\nAnswer: "},
  {"role": "assistant", "content": "Dry palms produce more friction and heat due to the absence of lubricants."},
  {"role": "assistant", "content": "Wet palms produce less heat because water reduces friction."},
  {"role": "assistant", "content": "Oil acts as a lubricant, reducing friction and heat."},
  {"role": "assistant", "content": "Lotion also reduces friction, similar to oil."},
  {"role": "assistant", "content": "The answer is: A"},
  {"role": "user", "content": "Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?\nChoices: A - The refrigerator door is smooth. B - The refrigerator door contains iron. C - The refrigerator door is a good conductor. D - The refrigerator door has electric wires in it.\nAnswer: "},
  {"role": "assistant", "content": "Smoothness does not affect magnetic properties, so this is unlikely to be correct."},
  {"role": "assistant", "content": "Magnets stick to iron and other ferromagnetic materials."},
  {"role": "assistant", "content": "Conductivity is not related to magnetism."},
  {"role": "assistant", "content": "Electric wires do not influence magnetic attraction."},
  {"role": "assistant", "content": "The answer is: B"},
  {"role": "user", "content": "Question: A fold observed in layers of sedimentary rock most likely resulted from the\nChoices: A - cooling of flowing magma. B - converging of crustal plates. C - deposition of river sediments. D - solution of carbonate minerals.\nAnswer: "},
  {"role": "assistant", "content": "Cooling of magma forms igneous rocks, not folds."},
  {"role": "assistant", "content": "Converging crustal plates cause pressure, forming folds."},
  {"role": "assistant", "content": "Deposition forms flat layers, not folds."},
  {"role": "assistant", "content": "Solution of carbonate minerals leads to dissolution, not folds."},
  {"role": "assistant", "content": "The answer is: B"},
  {"role": "user", "content": "Question: As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight and mass of the astronaut when standing on the Earth?\nChoices: A - 31 pounds and 14 kilograms B - 31 pounds and 84 kilograms C - 186 pounds and 14 kilograms D - 186 pounds and 84 kilograms\nAnswer: "},
  {"role": "assistant", "content": "31 pounds and 14 kilograms is incorrect because the astronaut's mass is 84 kilograms."},
  {"role": "assistant", "content": "31 pounds and 84 kilograms is incorrect due to the difference in gravity between the Moon and Earth."},
  {"role": "assistant", "content": "186 pounds and 14 kilograms is incorrect as the astronaut's mass is 84 kilograms."},
  {"role": "assistant", "content": "186 pounds and 84 kilograms is correct because the astronaut's mass remains 84 kilograms, and the weight on Earth would be 186 pounds."},
  {"role": "assistant", "content": "The answer is: D"},
  {"role": "user", "content": "Question: Some animals are very rare. For example, there are very few Siberian tigers. If the only Siberian tigers left are female, what will most likely happen?\nChoices: A - The females will find another type of male animal to mate with and produce more Siberian tigers. B - The females will mate with each other and produce more Siberian tigers. C - The females will only be able to produce female Siberian tigers. D - The females will not be able to produce more Siberian tigers, and they will die out.\nAnswer: "},
  {"role": "assistant", "content": "Females finding another type of male animal is unlikely as different species cannot mate and produce viable offspring."},
  {"role": "assistant", "content": "Females cannot mate with each other as female mammals cannot reproduce without male gametes."},
  {"role": "assistant", "content": "Females alone cannot produce any offspring without males."},
  {"role": "assistant", "content": "Females will not be able to produce more Siberian tigers, leading to their extinction."},
  {"role": "assistant", "content": "The answer is: D"},
  {"role": "user", "content": "Question: The following mathematical expressions represent four different concentrations of a chemical solution to be used in a science experiment. Which one is equal in magnitude to 1/1000?\nChoices: A - 1.0 x 10^3 B - 1.0 x 10^4 C - 1.0 x 10^-3 D - 1.0 x 10^-4\nAnswer: "},
  {"role": "assistant", "content": "1.0 x 10^3 equals 1000, not 1/1000."},
  {"role": "assistant", "content": "1.0 x 10^4 equals 10000, not 1/1000."},
  {"role": "assistant", "content": "1.0 x 10^-3 equals 0.001, which is 1/1000."},
  {"role": "assistant", "content": "1.0 x 10^-4 equals 0.0001, not 1/1000."},
  {"role": "assistant", "content": "The answer is: C"},
  {"role": "user", "content": "Question: Four materials are put into small containers. These materials are then moved from the small containers into larger containers. Which material will spread out to completely fill a larger container?\nChoices: A - air B - ice C - sand D - water\nAnswer: "},
  {"role": "assistant", "content": "Air spreads out to fill the entire volume of a larger container."},
  {"role": "assistant", "content": "Ice retains its shape and does not spread out."},
  {"role": "assistant", "content": "Sand only fills the bottom of the container."},
  {"role": "assistant", "content": "Water takes the shape of the container but does not fill the entire volume unless it is enough to do so."},
  {"role": "assistant", "content": "The answer is: A"},
  {"role": "user", "content": "Question: Which of these human activities does not contribute to the extinction of species?\nChoices: A - hunting B - habitat destruction C - restoration ecology D - introduced nonnative species\nAnswer: "},
  {"role": "assistant", "content": "Hunting reduces populations, contributing to extinction."},
  {"role": "assistant", "content": "Habitat destruction leads to loss of living spaces, contributing to extinction."},
  {"role": "assistant", "content": "Restoration ecology restores ecosystems, preventing extinction."},
  {"role": "assistant", "content": "Introduced nonnative species outcompete and harm native species, leading to extinction."},
  {"role": "assistant", "content": "The answer is: C"}
]
        return messages_arc
    elif dataset_name == "gsm8k":
        messages_gsm8k = []
        rand_list_from_train = np.random.choice(gsm8k_datalist, 9, replace=False)
        for data in rand_list_from_train:
            l = []
            d = {"role": "user", "content": "Question: " + data['question'] + "\nAnswer:"}
            l.append(d)
            data['answer'] = data['answer'].replace("####", "The answer is:")
            answers = data['answer'].split("\n")
            for answer in answers:
                if answer == "":
                    continue
                # temp_d = d.copy()
                if not answer.endswith("."):
                    answer = answer + "."
                # temp_d['content'] += " " + answer
                # l.append(temp_d)
                # d = temp_d.copy()
                l.append({"role": "assistant", "content": answer})
            messages_gsm8k.extend(l)
        return messages_gsm8k
    elif dataset_name == "math":
        messages_math = [{"role": "system", "content": "You are a smart assistant that solves math problems. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. If you think you're ready to output the answer, you can wrap your answer with \\boxed{}. Please follow this format metrically"}]
        rand_list_from_train = np.random.choice(math_datalist, 9, replace=False)
        for data in rand_list_from_train:
            l = []
            d = {"role": "user", "content": "Question: " + data['problem'] + "\nAnswer:"}
            l.append(d)
            answers = data['solution'].split(". ")
            for answer in answers:
                if answer == "":
                    continue
                l.append({"role": "assistant", "content": answer})
            messages_math.extend(l)
        return messages_math
    elif dataset_name == "ecqa":
        messages_ecqa = [{
            "role": "system",
            "content": "You are a smart assistant that answer multiple chocies. You will think about the question and generate the reasoning trajectory step by step. You will only generate ONE sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. There might be multiple candidates answers， but you should output the most likely one given the question. please finish the response with The answer is: <Answer>."
        }]
        ecqa_key_mapping = {}
        for line in open("/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/sampling/sampling_code/ecqa.jsonl", "r"):
            d = json.loads(line)
            ecqa_key_mapping[d['id']] = [d['positives'], d['negatives']]
        
        ecqa_train_dataset = np.random.choice(ecqa_datalist, 8, replace=False)
        
        for ecqa_data in ecqa_train_dataset:
            l = []
            d = "Question: " + ecqa_data['question'] + '\nChoices: '
            for i in range(len(ecqa_data['choices']['text'])):
                d += chr(ord('A') + i) + " - " + ecqa_data['choices']['text'][i] + " "
            d = d[:-1] + '\nAnswer:'
            l.append({"role": "user", "content": d})    
            positives = ' '.join(ecqa_key_mapping[ecqa_data['id']][0])
            negatives = ecqa_key_mapping[ecqa_data['id']][1]
            positive_insert_id = ord(ecqa_data['answerKey']) - ord('A')
            # insert the negative choice at the corrspoding position in positives
            negatives = negatives[:positive_insert_id] + [positives] + negatives[positive_insert_id:]
            for i in range(len(negatives)):
                l.append({"role": "assistant", "content": negatives[i]})
                # l.append({"role": "assistant", "content": 'Let\'s look at answer choice: ' + chr(ord('A') + i) + ', '+ negatives[i]})
            l.append({"role": "assistant", "content": ' The answer is: ' + ecqa_data['answerKey']})
            messages_ecqa.extend(l)
        return messages_ecqa
    elif dataset_name == "proofwriter":
        messages_proofwriter = [
            {"role": "system", "content": "You are a smart assistant that tries to answer a question. You will try to answer the question based on the context given to you. The answer is one of A, B or C, which represents True, False and Unknown. If you think you're ready to output the answer, you can finish the response with The answer is: <Answer>."}, 
            {"role": "user", "content": "Question: Based on the context information, is the following statement true, false, or unknown? The cat is not round.\nContext: The cat eats the squirrel. The cat sees the squirrel. The cow eats the squirrel. The cow sees the cat. The rabbit is round. The rabbit sees the cat. The squirrel eats the rabbit. The squirrel is cold. The squirrel needs the rabbit. The squirrel sees the cat. If someone sees the cat and they are not green then they see the cow. If the rabbit is kind and the rabbit sees the squirrel then the squirrel needs the rabbit. Rough people are cold. If someone sees the rabbit then they are not round. If someone sees the squirrel and they are not green then they need the squirrel. If someone eats the cow then they see the rabbit. If someone eats the squirrel then they are rough. If someone is cold then they eat the cow.\nChoices: A - True B - False C - Unknown\nAnswer: "},
            {"role": "assistant", "content": "The cat eats the squirrel."},
            {"role": "assistant", "content": "If someone eats the squirrel then they are rough."},
            {"role": "assistant", "content": "Since the cat eats the squirrel, the cat is rough."},
            {"role": "assistant", "content": "Rough people are cold."},
            {"role": "assistant", "content": "Since the cat is rough, the cat is cold."},
            {"role": "assistant", "content": "If someone sees the rabbit then they are not round."},
            {"role": "assistant", "content": "The cat sees the rabbit, so the cat is not round."},
            {"role": "assistant", "content": "The answer is: A"},
            {"role": "user", "content": "Question: Based on the above information, is the following statement true, false, or unknown? Harry is not green.\nContext: Anne is furry. Anne is nice. Anne is smart. Bob is young. Erin is nice. Harry is smart. Harry is young. Young things are furry. Nice, furry things are green. All green things are nice. Nice, green things are big. All green things are smart. If something is big and young then it is round. All green things are big. If Harry is young then Harry is furry. Furry, smart things are nice.\nChoices: A - True B - False C - Unknown\nAnswer: "},
            {"role": "assistant", "content": "Harry is young."},
            {"role": "assistant", "content": "Young things are furry."},
            {"role": "assistant", "content": "Since Harry is young, Harry is furry."},
            {"role": "assistant", "content": "Harry is smart."},
            {"role": "assistant", "content": "Furry, smart things are nice."},
            {"role": "assistant", "content": "Since Harry is furry and smart, Harry is nice."},
            {"role": "assistant", "content": "Nice, furry things are green."},
            {"role": "assistant", "content": "Since Harry is nice and furry, Harry is green."},
            {"role": "assistant", "content": "The answer is: B"},
            {"role": "user", "content": "Question: Based on the above information, is the following statement true, false, or unknown? The lion does not like the cat.\nContext: The cat chases the lion. The cat is rough. The cat is young. The cat needs the lion. The cat needs the rabbit. The dog is green. The dog is young. The dog likes the cat. The lion is blue. The lion is green. The rabbit chases the lion. The rabbit is blue. The rabbit is rough. The rabbit likes the dog. The rabbit needs the dog. The rabbit needs the lion. If someone chases the lion then they are round. If the lion needs the rabbit and the rabbit chases the dog then the lion likes the dog. If someone is round and they chase the lion then they need the cat. If someone needs the cat and they chase the dog then they like the rabbit. If someone chases the lion and the lion is blue then the lion is round. If someone chases the rabbit then they are rough. If someone is rough and they like the rabbit then the rabbit is young. If the rabbit chases the cat and the cat needs the lion then the rabbit is young. If someone is round and they need the cat then they chase the dog.\nChoices: A - True B - False C - Unknown\nAnswer: "},
            {"role": "assistant", "content": "The cat chases the lion."},
            {"role": "assistant", "content": "If someone chases the lion, then they are round."},
            {"role": "assistant", "content": "Since the cat chases the lion, the cat is round."},
            {"role": "assistant", "content": "If someone is round and they chase the lion, then they need the cat."},
            {"role": "assistant", "content": "Since the cat is round and chases the lion, the cat needs the cat."},
            {"role": "assistant", "content": "The given information does not specify whether the lion likes the cat."},
            {"role": "assistant", "content": "The answer is: C"},
            # {"role": "user", "content": "Question: Based on the above information, is the following statement true, false, or unknown? The tiger is not round.\nContext: The cow chases the tiger. The cow eats the rabbit. The cow is red. The cow is young. The cow sees the squirrel. The rabbit eats the cow. The rabbit is young. The squirrel is kind. The squirrel is red. The tiger is red. The tiger is young. The tiger sees the squirrel. If the cow sees the squirrel and the squirrel eats the rabbit then the rabbit sees the tiger. If something sees the tiger then the tiger is round. If something chases the squirrel then the squirrel eats the rabbit. If something is round then it sees the tiger. If something is big then it chases the squirrel. If something eats the rabbit then the rabbit chases the squirrel. If the tiger eats the cow and the cow sees the squirrel then the squirrel is young.\nChoices: A - True B - False C - Unknown\nAnswer: "},
            # {"role": "assistant", "content": "The cow chases the tiger."},
            # {"role": "assistant", "content": "The tiger sees the squirrel."},
            # {"role": "assistant", "content": "If something sees the tiger, then the tiger is round."},
            # {"role": "assistant", "content": "Since the tiger sees the squirrel, the  tiger sees the tiger."},
            # {"role": "assistant", "content": "If something is round, then it sees the tiger."},
            # {"role": "assistant", "content": "Since the tiger sees the squirrel and sees the tiger, the tiger is round."},
            # {"role": "assistant", "content": "The tiger is round, so the statement 'The tiger is not round' is false."},
            # {"role": "assistant", "content": "The answer is: B"},
            # {"role": "user", "content": "Question: Based on the above information, is the following statement true, false, or unknown? The mouse is not nice.\nContext: The cow likes the lion. The cow likes the mouse. The cow likes the squirrel. The lion eats the mouse. The lion is green. The lion likes the squirrel. The mouse is blue. The mouse likes the cow. The mouse visits the lion. The squirrel visits the cow. If someone visits the cow then they eat the mouse. If someone likes the mouse then they visit the lion. If someone visits the cow then they eat the cow. If someone visits the squirrel and the squirrel is blue then the squirrel likes the cow. If someone is cold then they eat the cow. If someone is green then they are big. If someone is big and green then they like the lion. If someone likes the lion then they are cold. If someone eats the cow then they visit the cow.\nChoices: A) True B) False C) Unknown\nAnswer: "},
            # {"role": "assistant", "content": "The mouse likes the cow."},
            # {"role": "assistant", "content": "If someone likes the mouse then they visit the lion."},
            # {"role": "assistant", "content": "Since the mouse likes the cow, the mouse visits the lion."},
            # {"role": "assistant", "content": "If someone visits the cow then they eat the mouse."},
            # {"role": "assistant", "content": "The squirrel visits the cow."},
            # {"role": "assistant", "content": "The given information does not specify whether the mouse is nice."},
            # {"role": "assistant", "content": "The answer is: C"},
            # {"role": "user", "content": "Question: Based on the above information, is the following statement true, false, or unknown? The squirrel eats the squirrel.\nContext: The dog eats the mouse. The dog eats the tiger. The dog visits the squirrel. The mouse is green. The mouse visits the tiger. The squirrel is big. The squirrel is round. The squirrel likes the dog. The tiger eats the dog. The tiger visits the mouse. If someone eats the tiger and the tiger is big then they are green. If someone is green then they like the squirrel. If the dog is green then the dog likes the mouse. If someone visits the tiger then the tiger is blue. If someone visits the tiger then the tiger visits the dog. If someone is blue and they eat the squirrel then the squirrel is green. If someone is blue then they eat the squirrel. If someone likes the dog and they are green then they are blue.\nChoices: A) True B) False C) Unknown\nAnswer: "},
            # {"role": "assistant", "content": "The squirrel is big and round."},
            # {"role": "assistant", "content": "The squirrel likes the dog."},
            # {"role": "assistant", "content": "The tiger visits the mouse."},
            # {"role": "assistant", "content": "If someone eats the tiger and the tiger is big, then they are green."},
            # {"role": "assistant", "content": "The dog eats the tiger, so if the tiger is big, the dog would be green."},
            # {"role": "assistant", "content": "If someone is green, they like the squirrel."},
            # {"role": "assistant", "content": "If someone likes the dog and they are green, then they are blue."},
            # {"role": "assistant", "content": "The statement 'The squirrel eats the squirrel' follows the given context."},
            # {"role": "assistant", "content": "The answer is: A"}
        ]
        return messages_proofwriter
    elif dataset_name == "mmlu_pro":
        messages_mmlu_pro = [
        {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: "},
  {"role": "assistant", "content": "Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \)."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \)."},
  {"role": "assistant", "content": "Let's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition."},
  {"role": "assistant", "content": "Let's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$."},
  {"role": "assistant", "content": "Let's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$.\nLet's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition."},
  {"role": "assistant", "content": "Let's think about answer choice E, 12. Since the characteristic should be smaller than 12, considering \(2 \cdot 2 = 4\) and \(6 \cdot 2 = 12\), it indicates 12 is too large."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$.\nLet's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition.\nLet's think about answer choice E, 12. Since the characteristic should be smaller than 12, considering \(2 \cdot 2 = 4\) and \(6 \cdot 2 = 12\), it indicates 12 is too large."},
  {"role": "assistant", "content": "Let's think about answer choice F, 50. A characteristic of 50 means \(50 \cdot 2 = 0\), but 50 is larger than necessary, so this is not the smallest number satisfying the condition."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$.\nLet's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition.\nLet's think about answer choice E, 12. Since the characteristic should be smaller than 12, considering \(2 \cdot 2 = 4\) and \(6 \cdot 2 = 12\), it indicates 12 is too large.\nLet's think about answer choice F, 50. A characteristic of 50 means \(50 \cdot 2 = 0\), but 50 is larger than necessary, so this is not the smallest number satisfying the condition."},
  {"role": "assistant", "content": "Let's think about answer choice G, 2. For the ring $2\mathbb{Z}$, the characteristic should be 2 because \(2 \cdot 2 = 4\), but considering the logic where \( 2k = 0 \) only for \( k = 0 \), the characteristic can be seen as 0."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$.\nLet's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition.\nLet's think about answer choice E, 12. Since the characteristic should be smaller than 12, considering \(2 \cdot 2 = 4\) and \(6 \cdot 2 = 12\), it indicates 12 is too large.\nLet's think about answer choice F, 50. A characteristic of 50 means \(50 \cdot 2 = 0\), but 50 is larger than necessary, so this is not the smallest number satisfying the condition.\nLet's think about answer choice G, 2. For the ring $2\mathbb{Z}$, the characteristic should be 2 because \(2 \cdot 2 = 4\), but considering the logic where \( 2k = 0 \) only for \( k = 0 \), the characteristic can be seen as 0."},
  {"role": "assistant", "content": "Let's think about answer choice H, 100. Similar to the larger numbers we've analyzed, 100 is larger than necessary for the characteristic of $2\mathbb{Z}$, so it is not the smallest number."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$.\nLet's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition.\nLet's think about answer choice E, 12. Since the characteristic should be smaller than 12, considering \(2 \cdot 2 = 4\) and \(6 \cdot 2 = 12\), it indicates 12 is too large.\nLet's think about answer choice F, 50. A characteristic of 50 means \(50 \cdot 2 = 0\), but 50 is larger than necessary, so this is not the smallest number satisfying the condition.\nLet's think about answer choice G, 2. For the ring $2\mathbb{Z}$, the characteristic should be 2 because \(2 \cdot 2 = 4\), but considering the logic where \( 2k = 0 \) only for \( k = 0 \), the characteristic can be seen as 0.\nLet's think about answer choice H, 100. Similar to the larger numbers we've analyzed, 100 is larger than necessary for the characteristic of $2\mathbb{Z}$, so it is not the smallest number."},
  {"role": "assistant", "content": "Let's think about answer choice I, 20. Again, 20 is larger than necessary for the characteristic, so it's not the smallest number \( n \) such that \( n \cdot 2 = 0 \) in $2\mathbb{Z}$."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$.\nLet's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition.\nLet's think about answer choice E, 12. Since the characteristic should be smaller than 12, considering \(2 \cdot 2 = 4\) and \(6 \cdot 2 = 12\), it indicates 12 is too large.\nLet's think about answer choice F, 50. A characteristic of 50 means \(50 \cdot 2 = 0\), but 50 is larger than necessary, so this is not the smallest number satisfying the condition.\nLet's think about answer choice G, 2. For the ring $2\mathbb{Z}$, the characteristic should be 2 because \(2 \cdot 2 = 4\), but considering the logic where \( 2k = 0 \) only for \( k = 0 \), the characteristic can be seen as 0.\nLet's think about answer choice H, 100. Similar to the larger numbers we've analyzed, 100 is larger than necessary for the characteristic of $2\mathbb{Z}$, so it is not the smallest number.\nLet's think about answer choice I, 20. Again, 20 is larger than necessary for the characteristic, so it's not the smallest number \( n \) such that \( n \cdot 2 = 0 \) in $2\mathbb{Z}$."},
  {"role": "assistant", "content": "Let's think about answer choice J, 5. While 5 is smaller than many other numbers, it does not satisfy \(5 \cdot 2 = 0\) in $2\mathbb{Z}$ because the smallest number \( n \) must directly cancel the factor of 2."},
  {"role": "user", "content": "Question: The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements. Find the characteristic of the ring $2\mathbb{Z}$.\nChoices: A - 0 B - 30 C - 3 D - 10 E - 12 F - 50 G - 2 H - 100 I - 20 J - 5\nAnswer: Let's think about answer choice A, 0. A characteristic of a ring \( R \) is \( n \) if the statement \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) implies that \( k \) is a multiple of \( n \). Assume that \( ka = 0 \) for all \( a \in 2\mathbb{Z} \) for some \( k \). In particular, \( 2k = 0 \). Hence, \( k = 0 \) and \( n = 0 \).\nLet's think about answer choice B, 30. The characteristic of $2\mathbb{Z}$ is the smallest positive integer \( n \) such that \( n \cdot 2 = 0 \), so 30 is too large and not the smallest number satisfying this condition.\nLet's think about answer choice C, 3. The characteristic of $2\mathbb{Z}$ should be related to 2, so 3 is not suitable because \(3 \cdot 2 \neq 0\) in $2\mathbb{Z}$.\nLet's think about answer choice D, 10. For $2\mathbb{Z}$, the characteristic should be the smallest number \( n \) such that \( n \cdot 2 = 0 \), and 10 is not the smallest number that satisfies this condition.\nLet's think about answer choice E, 12. Since the characteristic should be smaller than 12, considering \(2 \cdot 2 = 4\) and \(6 \cdot 2 = 12\), it indicates 12 is too large.\nLet's think about answer choice F, 50. A characteristic of 50 means \(50 \cdot 2 = 0\), but 50 is larger than necessary, so this is not the smallest number satisfying the condition.\nLet's think about answer choice G, 2. For the ring $2\mathbb{Z}$, the characteristic should be 2 because \(2 \cdot 2 = 4\), but considering the logic where \( 2k = 0 \) only for \( k = 0 \), the characteristic can be seen as 0.\nLet's think about answer choice H, 100. Similar to the larger numbers we've analyzed, 100 is larger than necessary for the characteristic of $2\mathbb{Z}$, so it is not the smallest number.\nLet's think about answer choice I, 20. Again, 20 is larger than necessary for the characteristic, so it's not the smallest number \( n \) such that \( n \cdot 2 = 0 \) in $2\mathbb{Z}$.\nLet's think about answer choice J, 5. While 5 is smaller than many other numbers, it does not satisfy \(5 \cdot 2 = 0\) in $2\mathbb{Z}$ because the smallest number \( n \) must directly cancel the factor of 2."},
  {"role": "assistant", "content": "The answer is: A"}]
        return messages_mmlu_pro

def get_normalized_answer(dataset_name, data):
    if dataset_name == "arc" or dataset_name == "ecqa":
        return data['answerKey']
    elif dataset_name == "gsm8k":
        return extract_and_convert_number_real(data['answer'].split("####")[1].strip())
    elif dataset_name == "math":
        return remove_boxed(last_boxed_only_string(data['solution']))
    elif dataset_name == "proofwriter":
        return data['answer']
    elif dataset_name == "mmlu_pro":
        return data['answer']
    
def get_dataset_key(dataset_name):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "gsm8k" or dataset_name == "mmlu_pro":
        return "question"
    elif dataset_name == "math":
        return "problem"
    elif dataset_name == "proofwriter":
        return "context"
    
def get_normalized_prediction(dataset_name, prediction):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "proofwriter" or dataset_name == "mmlu_pro":
        normalized_prediction = prediction.strip().replace(": ", "").split(' ')[0]
        return normalized_prediction
    elif dataset_name == "gsm8k":
        return extract_and_convert_number_real(prediction)
    elif dataset_name == "math":
        return remove_boxed(last_boxed_only_string(prediction))

def get_list_of_end_prompts(dataset_name):
    if dataset_name == "arc" or dataset_name == "ecqa" or dataset_name == "gsm8k" or dataset_name == "proofwriter" or dataset_name == "mmlu_pro":
        return ['the answer is', 'The correct answer is', 'The answer is', 'The answer is indeed', 'the correct answer is indeed', 'The correct answer is indeed']
    elif dataset_name == "math":
        return ['boxed']

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
                "seed": 14,
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
            "seed": 14,
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
        # perplexity_with_rationale_list = [item * 0.9 ** i for i, item in enumerate(perplexity_with_rationale_list)]
        perplexity_with_rationale_list = [item * 1 ** i for i, item in enumerate(perplexity_with_rationale_list)]

        prob_list.append(sum(perplexity_with_rationale_list) / len(perplexity_with_rationale_list))
        if debug:
            print("Response: " + response)
            print("Perplexity: " + str(prob_list[-1]))
    return responses[np.argmax(prob_list)]

async def get_response(data, pbar: tqdm, heuristic: str, agent_model: str, world_model: str, agent_url: str, dataset: str):
    previous = get_previous(dataset, data)
    prediction, response_list = "", [""]
    normalized_answer, normalized_prediction = "", ""
    new_messages = get_messages(dataset).copy()
    new_messages.append({
        "role": "user",
        "content": previous
    })
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
        if response_list[chosen_previous] != "":
        # if True:
            new_messages.append({
                "role": "assistant",
                "content": response_list[chosen_previous]
            })
        # special case for when the heuristic is ''world_model_content'', the rationales are used to help with the generation of next action
        if chosen_world_model_response is not None and chosen_world_model_response.strip() != "We are ready to output the final answer":
            if dataset == "ecqa":
                previous_temp = previous + " <BOT>" + chosen_world_model_response + ". Specifically, let\'s look at answer choice " + chr(ord('A') + trail) + "<EOT>"
            else:
                previous_temp = previous + " <BOT>" + chosen_world_model_response + "<EOT>"
        # if debug:
        #     print("Chosen response: " + str(chosen_previous))
        
        
        url = get_url()
        content = {
            "model": agent_model,
            "messages": new_messages,
            "max_tokens": 1000,
            "temperature": 0.7,
            "stop_token_ids": [128001, 128009],
            "best_of": num_choices,
            "n": num_choices,
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
            
            for term in get_list_of_end_prompts(dataset):
                if term in previous:
                    try:
                        if dataset != "math":
                            prediction = previous.split(term)[1].replace('\n', ' ').strip()
                        else:
                            prediction = previous
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
        "question": data[get_dataset_key(dataset)],
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
    parser.add_argument("--write_file", type=str, default=None, help="File to write the output to")
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
    if not args.write_file:
        write_file = open("output_" + dataset + ".jsonl", 'w')
    else:
        write_file = open(args.write_file, 'w')
    debug = args.debug
    
    data_list = setup_datalist(args.dataset)
    new_messages = get_messages(dataset).copy()
    print("prompt length: ", len(tokenizer.apply_chat_template(new_messages, tokenize=False)))
    
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
            print("The answer is: ", get_normalized_answer(dataset, data_list[i]))
            apply_async([data_list[i]], heuristic, agent_model, world_model, agent_url, dataset)
        exit()
    
    chunks = [data_list[i:i + 2000] for i in range(0, len(data_list), 2000)]
    for chunk in chunks:
        result = apply_async(chunk, heuristic, agent_model, world_model, agent_url, dataset)
        for d in result:
            write_file.write(json.dumps(d) + '\n')
    write_file.close()
    print("Total TIME: ", time.time() - start_time)
