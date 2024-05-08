import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import json
import time
import numpy as np

# specify how to quantize the model
# quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype="torch.float16",
# )

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
start_time = time.time()

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
    {
        "role": "user",
        "content": "Question: He came across a raw item but his pack was full, he had to abandon something if he was to what the item?\nChoices: A - join, B - acquire, C - engage, D - maintain, E - remit\nAnswer: Acquiring an item requires one to have space to carry it. As he had box full of items, he had to abandon one of them in order to acquire new one. Join and Engage is not related to item. Maintain an item does not require one to abandon existing item. One cannot remit raw item. The answer is B - acquire."
    },
    {
        "role": "assistant",
        "content": "Question: He came across a raw item but his pack was full, he had to abandon something if he was to what the item?\nChoices: A - join, B - acquire, C - engage, D - maintain, E - remit\nAnswer: Acquiring an item requires one to have space to carry it. <BOT>We also need to establish relationship between acquire and the question itself<EOT>As he had box full of items, he had to abandon one of them in order to acquire new one. <BOT>Let's check whether other answer choices are related to the word item in question<EOT>Join and Engage is not related to item. <BOT>Let's check whether other answer choices are contradictory to the word abandon in question<EOT>Maintain an item does not require one to abandon existing item. <BOT>Let's check whether other answer choices are related to the word raw item in question<EOT>One cannot remit raw item. <BOT>We are ready to make final prediction<EOT>The answer is B - acquire."
    },
    {
        "role": "user",
        "content": "Question: Where is aberdeen in the US located?\nChoices: A - washington, B - europe, C - scotland, D - maryland, E - south dakota\nAnswer: Aberdeen is located in Washington state which is in US. Aberdeen is also located in Scotland which is part of Europe. However, Scotland or Europe are not inside US. Aberdeen is not located in Maryland and South Dakota states. The answer is A - washington."
    },
    {
        "role": "assistant",
        "content": "Question: Where is aberdeen in the US located?\nChoices: A - washington, B - europe, C - scotland, D - maryland, E - south dakota\nAnswer: Aberdeen is located in Washington state which is in US. <BOT>We need to think whether there is another aberdeen with the same name<EOT>Aberdeen is also located in Scotland which is part of Europe. <BOT>The answer choice need to be in the U.S. as mentioned in the question<EOT>However, Scotland or Europe are not inside US. <BOT>Let's check whether aberdeen is located in other answer choices<EOT>Aberdeen is not located in Maryland and South Dakota states. <BOT>We are ready to make final prediction<EOT>The answer is A - washington."
    },
    {
        "role": "user",
        "content": "Question: What has an accelerator and is owned by most people?\nChoices: A - vehical, B - fuel system, C - accelerate, D - airplane, E - car\nAnswer: Car is owned by most people. Car has an acclerator. Fuel system does not have accelerator. Aeroplanes are not owned by most people. Accelerate is not an entity which can be owned by people. Vehicle also include some types like bikes which does not have accelerator. The answer is E - car."
    },
    {
        "role": "assistant",
        "content": "Question: What has an accelerator and is owned by most people?\nChoices: A - vehical, B - fuel system, C - accelerate, D - airplane, E - car\nAnswer: Car is owned by most people. <BOT>We need to establish relationship between answer choice car and the word accelerator in question<EOT>Car has an acclerator. <BOT>Let's check whether other answer choices are related to the word accelerator in question<EOT>Fuel system does not have accelerator. <BOT>Let's check whether other answer choices are related to the word owned by most people in question<EOT>Aeroplanes are not owned by most people. <BOT>Let's check whether other answer choices are related to the word owned by most people in question<EOT>Accelerate is not an entity which can be owned by people. <BOT>Let's check whether other answer choices don't necessary contain an accelerator<EOT>Vehicle also include some types like bikes which does not have accelerator. <BOT>We are ready to make final prediction<EOT>The answer is E - car."
    },
]


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['train'])
write_file = open("llama3_output_with_score.txt", "w")
for i in range(len(data_list)):
    new_messages = messages_start_rationale.copy()
    new_messages.append({
        "role": "user",
        "content": "Question: " + data_list[i]['question'] + "\nAnswer: " + data_list[i]['answer']
    })
    input_ids = tokenizer.apply_chat_template(
        new_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=3000,
        eos_token_id=terminators,
        temperature=0.0,
        return_dict_in_generate=True,
        output_scores=True
    )
    response = tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
    transition_scores = model.compute_transition_scores(
        outputs.sequences, outputs.scores, normalize_logits=True
    )
    input_length = input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]

    score_string = ""
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | logits | probability
        score_string += f"|| {tokenizer.decode(tok):8s} | {score.cpu().numpy():.4f} "
    
    d = {"input": "Question: " + data_list[i]['question'] + "\nAnswer: " + data_list[i]['answer'], "output": response, "scores": score_string}
    # print(response)
    write_file.write(json.dumps(d) + "\n")
    write_file.write("----------------------------------------\n")

