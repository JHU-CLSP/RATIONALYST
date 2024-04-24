import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import json
import time
import numpy as np


model = AutoModelForCausalLM.from_pretrained('output/', device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("output")
start_time = time.time()

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
    {
        "role": "user",
        # "content": "Question: At 30, Anika is 4/3 the age of Maddie. What would be their average age in 15 years?\nAnswer: If Anika is 30 now, in 15 years, she'll be 30+15=<<30+15=45>>45 years old.  At 30, Anika is 4/3 the age of Maddie, meaning Maddie is 4/3*30=<<4/3*30=40>>40 years.  In 15 years, Maddie will be 40+15=<<40+15=55>>55 years old.  Their total age in 15 years will be 55+45=<<55+45=100>>100",
        "content": "Question: At 30, Anika is 4/3 the age of Maddie. What would be their average age in 15 years?\nAnswer: If Anika is 30 now, in 15 years, she'll be 30+15=<<30+15=45>>45 years old.  At 30, Anika is 4/3 the age of Maddie, meaning Maddie is 4/3*30=<<4/3*30=40>>40 years.  In 15 years, Maddie will be 40+15=<<40+15=55>>55 years old."
    },
]


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


input_ids = tokenizer.apply_chat_template(
    messages_start_rationale,
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
print(response)
