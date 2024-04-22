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
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72",
    },
    {
        "role": "assistant",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n <BOT>Let's first calculate the amount of chips Natalia sold in May<EOT> Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. <BOT>We have already calculated the number of chips in May, now we should calculate the sum of chips in April and May<EOT> Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72",
    },
    {
        "role": "user",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
    },
    {
        "role": "assistant",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: <BOT>To find out how much Weng earns per minute, we need to divide her hourly rate by the number of minutes in an hour<EOT> Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. <BOT>Now, to find out her total earnings for 50 minutes, we multiply the rate per minute by the number of minutes she worked<EOT> Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5"
    },
    {
        "role": "assistant",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: <BOT>To determine how much money Betty initially has, we calculate half of the total wallet cost<EOT> In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. <BOT>Betty’s grandparents contribute twice what her parents gave, so we need to calculate this amount<EOT> Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. <BOT>To find out how much more Betty needs to save, we subtract the total contributions and her initial amount from the total cost of the wallet<EOT> This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5"
    }
]
messages = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
    {
        "role": "user",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72",
    },
    {
        "role": "assistant",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\n Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. <BOT>We have already calculated the number of chips in May, now we should calculate the sum of chips in April and May<EOT> Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72",
    },
    {
        "role": "user",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
    },
    {
        "role": "assistant",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. <BOT>Now, to find out her total earnings for 50 minutes, we multiply the rate per minute by the number of minutes she worked<EOT> Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
    },
    {
        "role": "user",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5"
    },
    {
        "role": "assistant",
        "content": "Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?\nAnswer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. <BOT>Betty’s grandparents contribute twice what her parents gave, so we need to calculate this amount<EOT> Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. <BOT>To find out how much more Betty needs to save, we subtract the total contributions and her initial amount from the total cost of the wallet<EOT> This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5"
    }
]


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['train'])
write_file = open("llama3_output_with_score.txt", "w")
for i in range(len(data_list)):
    new_messages = messages.copy()
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

