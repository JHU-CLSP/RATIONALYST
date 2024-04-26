import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import json
import time
import numpy as np


agent_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
agent_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
agent_tokenizer.pad_token = agent_tokenizer.eos_token

messages_gsm8k = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math word problems. You will think step-by-step and only generate one sentence given the question and previous context. If you think you're ready to output the final answer, you can finish the response with The answer is:"
    },
    {
        "role": "user",
        "content": "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?"
    },
    {
        "role": "assistant",
        "content": "Answer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6"
    },
    {
        "role": "user",
        "content": "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
    },
    {
        "role": "assistant",
        "content": "Answer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5"
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
    },
    {
        "role": "assistant",
        "content": "Answer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39"
    },
    {
        "role": "user",
        "content": "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?"
    },
    {
        "role": "assistant",
        "content": "Answer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8"
    },
    {
        "role": "user",
        "content": "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?"
    },
    {
        "role": "assistant",
        "content": "Answer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9"
    },
    {
        "role": "user",
        "content": "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?"
    },
    {
        "role": "assistant",
        "content": "Answer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29"
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?"
    },
    {
        "role": "assistant",
        "content": "Answer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33"
    },
    {
        "role": "user",
        "content": "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
    },
    {
        "role": "assistant",
        "content": "Answer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8"
    }
]

terminators = [
    agent_tokenizer.eos_token_id,
    agent_tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['test'])
write_file = open("llama3_output_inference_cot.txt", "w")

total, correct = 0, 0
# for each question in the test set of GSM8K
for i in range(len(data_list)):
    total += 1
    question = "Question: " + data_list[i]['question']
    answer = data_list[i]['answer'].split("####")[1].strip()
    new_messages = messages_gsm8k.copy()
    new_messages.append({
        "role": "user",
        "content": question
    })
    input_ids = agent_tokenizer.apply_chat_template(
        new_messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(agent_model.device)
    attention_mask = torch.ones_like(input_ids)
    outputs = agent_model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=3000,
        eos_token_id=terminators,
        # temperature=0.1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=agent_tokenizer.eos_token_id
    )
    response = agent_tokenizer.decode(outputs.sequences[0][input_ids.shape[1]:], skip_special_tokens=True)
    input_length = input_ids.shape[1]
    generated_tokens = outputs.sequences[:, input_length:]
    print("response: " + response)
    try:
        prediction = response.split("The answer is")[1].strip()
        if prediction == answer:
            correct += 1
        print("final_answer: " + str(prediction) + '\t' + str(answer))
    except:
        print("Error")
        
print("Accuracy: ", correct/total)
print("Total: ", total)
print("Correct: ", correct)
    
