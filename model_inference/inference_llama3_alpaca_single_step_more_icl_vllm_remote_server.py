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


def extract_and_convert_number_real(text):
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

# agent_model = LLM(
#     model="meta-llama/Meta-Llama-3-8B-Instruct",
#     trust_remote_code=True,
#     tensor_parallel_size=1,
# )
# tokenizer = agent_model.get_tokenizer()

messages_gsm8k = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math word problems. You will only generate one sentence that extends the reasoning trajectory that solves the question given the question and partial answer reasoning trajectory. Please don't repeat your previous generation while you're generating the sentence. If you think you're ready to output the answer, you can finish the response with The answer is:"
    },
]

ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['test'])

train_data_list = list(ds['train'])
np.random.seed(14)
rand_list_from_train = np.random.choice(train_data_list, 10, replace=False)
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
    
# unanswered = []
# for line in open("llama3_output_inference_single_step.jsonl", "r"):
#     data = json.loads(line)
#     if data['prediction'] == "":
#         unanswered.append(data['question'])

# new_data_list = []
# for data in data_list:
#     if data['question'] in unanswered:
#         new_data_list.append(data)
# data_list = new_data_list

write_file = open("llama3_output_inference_single_step_unanswered_more_icl_vllm_new_version.jsonl", "w")
correct, has_answer, total = 0, 0, 0
start_time = time.time()

# for each question in the test set of GSM8K
for i in range(len(data_list)):
    total += 1
    previous = "Question: " + data_list[i]['question'] + "\nAnswer:"
    prediction = ""
    normalized_answer, normalized_prediction = "", ""
    answer = data_list[i]['answer'].split("####")[1].strip()
    for trail in range(15):
        new_messages = messages_gsm8k.copy()
        new_messages.append({
            "role": "user",
            "content": previous
        })
        url = 'http://c007:1234/v1/chat/completions'
        data = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": new_messages,
            "max_tokens": 1000,
            "temperature": 0,
            "stop_token_ids": [128001, 128009]
        }
        headers = {
            "Content-Type": "application/json"
        }
        world_response = requests.post(url, headers=headers, json = data)
        try:
            response = world_response.json()['choices'][0]['message']['content']
        except:
            print("Error in response, choice is empty")
            break
        
        print("Previous: " + previous)
        print("Partial response: " + response)
        
        if not response.endswith('.'):
            response = response + '.'
        previous = previous + ' ' + response
        
        if "The answer is:" in previous:
            try:
                prediction = response.split("The answer is:")[1].replace('\n', ' ').strip()
                normalized_prediction = extract_and_convert_number_real(prediction)
                normalized_answer = extract_and_convert_number_real(answer)
                if normalized_prediction == "":
                    print("Prediction is empty")
                else:
                    has_answer += 1
                    if normalized_prediction == normalized_answer:
                        correct += 1
                    print("final_answer: " + str(prediction) + '\t' + str(answer))
                    break
            except:
                print("Error")
            
        if trail == 14:
            print("Stuck in loop")
            break
    d = {
        "question": data_list[i]['question'],
        "answer": answer,
        "prediction": prediction,
        "normalized_answer": normalized_answer,
        "normalized_prediction": normalized_prediction,
        "full_response": previous.split("Answer:")[1].strip(),
    }
    write_file.write(json.dumps(d) + '\n')

print("Accuracy: " + str(correct/total))
print("Total: ", total)
print("Has answer: ", has_answer)
print("Correct: ", correct)
print("Time: ", time.time() - start_time)
