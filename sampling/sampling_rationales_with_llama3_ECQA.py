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
        "content": "Question: He came across a raw item but his pack was full, he had to abandon something if he was to what the item?\nChoices: A - join, B - acquire, C - engage, D - maintain, E - remit\nAnswer: Join is not the type of the activity associated with item. Acquiring requires space in a pack. To create space in a full pack, one has to abandon an item. Engage is not related to item. Maintain an item does not require one to make space for it. One cannot remit raw item. The answer is B - acquire."
    },
    {
        "role": "assistant",
        "content": "Question: He came across a raw item but his pack was full, he had to abandon something if he was to what the item?\nChoices: A - join, B - acquire, C - engage, D - maintain, E - remit\nAnswer: <BOT>Let's look at choice A to see whether it's related to item<EOT> Join is not the type of the activity associated with item. <BOT>Let's look at choice B to check out its meaning<EOT> Acquiring requires space in a pack. <BOT>Let's see whether the meaning of choice B is related to the question<EOT> To create space in a full pack, one has to abandon an item. <BOT>Let's look at choice C to see whether it's related to item<EOT> Engage is not related to item. <BOT>Let's look at choice D to check whether it's related to making spaces<EOT> Maintain an item does not require one to make space for it. <BOT>Let's look at choice E to check whether it's related to raw item in the question<EOT> One cannot remit raw item. The answer is B - acquire."
    },
    {
        "role": "user",
        "content": "Question: Where is aberdeen in the US located?\nChoices: A - washington, B - europe, C - scotland, D - maryland, E - south dakota\nAnswer: Aberdeen is located in Washington State. Washington state is located in US. Europe is not located in US. Scotland is not located in US. Aberdeen is not located in Maryland. Aberdeen is not located in South Dakota. The answer is A - washington." 
    },
    {
        "role": "assistant",
        "content": "Question: Where is aberdeen in the US located?\nChoices: A - washington, B - europe, C - scotland, D - maryland, E - south dakota\nAnswer: <BOT>Let's look at choice A to see whether it fulfills the requirement in question that aberdeen needs to be in this place<EOT> Aberdeen is located in Washington State. <BOT>Let's look at choice A to see whether it fulfills the requirement in question that the place needs to be in the US<EOT> Washington state is located in US. <BOT>Let's look at choice B to see whether it fulfills the requirement in question that the place needs to be in the US<EOT> Europe is not located in US. <BOT>Let's look at choice C to see whether it's fulfills the requirement in question that the place needs to be in the US<EOT> Scotland is not located in US. <BOT>Let's look at choice D to see whether it fulfills the requirement in question that aberdeen needs to be in this place<EOT> Aberdeen is not located in Maryland. <BOT>Let's look at choice E to see whether it fulfills the requirement in question that aberdeen needs to be in this place<EOT> Aberdeen is not located in South Dakota."
    },
    {
        "role": "user",
        "content": "Question: What has an accelerator and is owned by most people?\nChoices: A - vehical, B - fuel system, C - accelerate, D - airplane, E - car\nAnswer: Vehicle can contain some types like bikes which does not have accelerator. Fuel system does not have an accelerator. Accelerate is not an entity. Aeroplanes are not owned by most people. Car is owned by most people. Car has an accelerator. The answer is E - car."
    },
    {
        "role": "assistant",
        "content": "Question: What has an accelerator and is owned by most people?\nChoices: A - vehical, B - fuel system, C - accelerate, D - airplane, E - car\nAnswer: <BOT>Let's look at choice A to see whether it has an accelerator<EOT> Vehicle can contain some types like bikes which does not have accelerator. <BOT>Let's look at choice B to see whether it has an accelerator<EOT> Fuel system does not have an accelerator. <BOT>Let's look at choice C to see whether it has an accelerator<EOT> Accelerate is not an entity. <BOT>Let's look at choice D to see whether it is owned by most people<EOT> Aeroplanes are not owned by most people. <BOT>Let's look at choice E to see whether it is owned by most people<EOT> Car is owned by most people. <BOT>Let's look at choice E to see whether it has an accelerator<EOT> Car has an accelerator. The answer is E - car."
    },
    {
        "role": "user",
        "content": "Question: The fox walked from the city into the forest, what was it looking for?\nChoices: A - pretty flowers., B - hen house, C - natural habitat, D - storybook, E - dense forest\nAnswer: Foxes don't look for pretty flowers. Hen houses are not found in a forest. Foresets are one of the main natural habitats of foxes. Foxes don't look for storybooks. Dense forest is not the only part of forest inhabited by foxes. The answer is C - natural habitat."
    },
    {
        "role": "assistant",
        "content": "Question: The fox walked from the city into the forest, what was it looking for?\nChoices: A - pretty flowers., B - hen house, C - natural habitat, D - storybook, E - dense forest\nAnswer: <BOT>Let's look at choice A to see whether it's something that foxes do<EOT> Foxes don't look for pretty flowers. <BOT>Let's look at choice B to see whether it can be found in forest<EOT> Hen houses are not found in a forest. <BOT>Let's look at choice C to see whether it's related to foxes and forest<EOT> Foresets are one of the main natural habitats of foxes. <BOT>Let's look at choice D to see whether it's something that foxes do<EOT> Foxes don't look for storybooks. <BOT>Let's look at choice E to see whether it's an overspecification of forest<EOT> Dense forest is not the only part of forest inhabited by foxes. The answer is C - natural habitat."
    },
    {
        "role": "user",
        "content": "Question: What is the responsibility of an adult?\nChoices: A - drink beer, B - drive train, C - sweeping the floor, D - work, E - marry\nAnswer: Drinking beer is not a responsibility of every adult. Driving train is not responsibility of every adult. Every adult is not expected to sweep the floor. Every adult is not expected to get married. An adult person is expected to work and provide a living to his family. The answer is D - work."
    },
    {
        "role": "assistant",
        "content": "Question: What is the responsibility of an adult?\nChoices: A - drink beer, B - drive train, C - sweeping the floor, D - work, E - marry\nAnswer: <BOT>Let's look at choice A to see whether additive behaviors like drinking beer it's a responsibility of an adult<EOT> Drinking beer is not a responsibility of every adult. <BOT>Let's look at choice B to see whether specific work like driving train is a responsibility of every adult<EOT> Driving train is not responsibility of every adult. <BOT>Let's look at choice C to see whether household like sweeping floor it's a responsibility of every adult<EOT> Every adult is not expected to sweep the floor. <BOT>Let's look at choice D to see whether mature stuff like providing living to family it a responsibility of an adult<EOT> An adult person is expected to work and provide a living to his family. <BOT>Let's look at choice E to see whether personal choice like marriage is a responsibility of an adult<EOT> Every adult is not expected to get married. The answer is D - work."
    },
    {
        "role": "user",
        "content": "Question: Where could you find a very large amount of air?\nChoices: A - park, B - surface of earth, C - train station, D - space shuttle, E - house\nAnswer: Park is on the surface of earth. Train station is on the surface of earth. Surface of earth is all the area on the planet earth. Earth is covered by various gases which constitutes air. Space shuttle is not very big compared to surface of earth. House is on the surface of earth. The answer is B - surface of earth."
    },
    {
        "role": "assistant",
        "content": "Question: Where could you find a very large amount of air?\nChoices: A - park, B - surface of earth, C - train station, D - space shuttle, E - house\nAnswer: <BOT>Let's look at choice A to see whether it's part of other answer choices and other answer choices contains more air<EOT> Park is on the surface of earth. <BOT>Let's look at choice B to see whether it's part of other answer choices and other answer choices contains more air<EOT> Train station is on the surface of earth. <BOT>Let's look at choice C to understand what it is<EOT> Surface of earth is all the area on the planet earth. <BOT>Let's look at choice C to understand whether it contains a very large amount of air<EOT> Earth is covered by various gases which constitutes air. <BOT>Let's look at choice D to see other answer choices contains more air<EOT> Space shuttle is not very big compared to surface of earth. <BOT>Let's look at choice E to see whether it's part of other answer choices and other answer choices contains more air<EOT> House is on the surface of earth. The answer is B - surface of earth."
    },
    {
        "role": "user",
        "content": "Question: John loves animals and he hates animal abuse. Because of this, john is very careful about the places he goes. Where might he avoid going?\nChoices: A - pet store, B - parking lot, C - woodland, D - north america, E - park\nAnswer: Pet store is a shop selling enslaved and exploited animals as pets. Enslavement and exploitation of animals is animal abuse. Parking lot is an area where vehicles are parked. Animals may not be abused in parking lots. Woodland is a land covered with trees. Animals may not be abused in woodland. North America is a continent and has pet stores. All of North America does not abuse animals. Park is a large public garden. Animals may not be abused in parks. The answer is A - pet store."
    },
    {
        "role": "assistant",
        "content": "Question: John loves animals and he hates animal abuse. Because of this, john is very careful about the places he goes. Where might he avoid going?\nChoices: A - pet store, B - parking lot, C - woodland, D - north america, E - park\nAnswer: <BOT>Let's look at choice A to understand what pet store is<EOT> Pet store is a shop selling enslaved and exploited animals as pets. <BOT>Let's look at choice A to see whether it's related to animal abuse<EOT> Enslavement and exploitation of animals is animal abuse. <BOT>Let's look at choice B to see whether it's related to animal abuse<EOT> Parking lot is an area where vehicles are parked. Animals may not be abused in parking lots. <BOT>Let's look at choice C to see whether it's related to animal abuse<EOT> Woodland is a land covered with trees. Animals may not be abused in woodland. <BOT>Let's look at choice D to see whether it's related to animal abuse<EOT> North America is a continent and has pet stores. <BOT>Let's look at choice D to see whether it's an overgeneralization<EOT> All of North America does not abuse animals. <BOT>Let's look at choice E to understand what a park is<EOT> Park is a large public garden. <BOT>Let's look at choice E to see whether it's related to animal abuse<EOT> Animals may not be abused in parks. The answer is A - pet store."
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

