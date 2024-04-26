import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import datasets
import json
import time
import numpy as np
import vllm
from vllm import LLM
from vllm import SamplingParams

# world_model = AutoModelForCausalLM.from_pretrained('/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model/model_training/output/', device_map="auto")
# world_tokenizer = AutoTokenizer.from_pretrained("/weka/scratch/djiang21/Dongwei_quiet_star/reasoning_world_model//model_training/output/")
# world_tokenizer.pad_token = world_tokenizer.eos_token
# agent_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", device_map="auto")
# agent_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
# agent_tokenizer.pad_token = agent_tokenizer.eos_token

agent_model = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=1,
)
tokenizer = agent_model.get_tokenizer()

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
]

messages_gsm8k = [
    {
        "role": "system",
        "content": "You are a smart assistant that solves math word problems. You will think step-by-step and only generate one sentence given the question and previous context. If you think you're ready to output the final answer, you can finish the response with The answer is: number, where number is the answer to the question. Please make sure to stick to the answer output format."
    },
    {
        "role": "user",
        "content": "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer:",
    },
    {
        "role": "assistant",
        "content": "There are 15 trees originally."
    },
    {
        "role": "user",
        "content": "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer: There are 15 trees originally."
    },
    {
        "role": "assistant",
        "content": "Then there were 21 trees after some more were planted."
    },
    {
        "role": "user",
        "content": "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer: There are 15 trees originally. Then there were 21 trees after some more were planted."
    },
    {
        "role": "assistant",
        "content": "So there must have been 21 - 15 = 6."
    },
    {
        "role": "user",
        "content": "Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nAnswer: There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6."
    },
    {
        "role": "assistant",
        "content": "The answer is: 6"
    },
    {
        "role": "user",
        "content": "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "There are originally 3 cars."
    },
    {
        "role": "user",
        "content": "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer: There are originally 3 cars."
    },
    {
        "role": "assistant",
        "content": "2 more cars arrive."
    },
    {
        "role": "user",
        "content": "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer: There are originally 3 cars. 2 more cars arrive."
    },
    {
        "role": "assistant",
        "content": "3 + 2 = 5."
    },
    {
        "role": "user",
        "content": "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nAnswer: There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5."
    },
    {
        "role": "assistant",
        "content": "The answer is: 5"
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Originally, Leah had 32 chocolates."
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Originally, Leah had 32 chocolates."
    },
    {
        "role": "assistant",
        "content": "Her sister had 42."
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Originally, Leah had 32 chocolates. Her sister had 42."
    },
    {
        "role": "assistant",
        "content": "So in total they had 32 + 42 = 74."
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74."
    },
    {
        "role": "assistant",
        "content": "After eating 35, they had 74 - 35 = 39."
    },
    {
        "role": "user",
        "content": "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nAnswer: Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39."
    },
    {
        "role": "assistant",
        "content": "The answer is: 39"
    },
    {
        "role": "user",
        "content": "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer:",
    },
    {
        "role": "assistant",
        "content": "Jason started with 20 lollipops."
    },
    {
        "role": "user",
        "content": "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: Jason started with 20 lollipops."
    },
    {
        "role": "assistant",
        "content": "Then he had 12 after giving some to Denny."
    },
    {
        "role": "user",
        "content": "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny."
    },
    {
        "role": "assistant",
        "content": "So he gave Denny 20 - 12 = 8."
    },
    {
        "role": "user",
        "content": "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nAnswer: Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8."
    },
    {
        "role": "assistant",
        "content": "The answer is: 8"
    },
    {
        "role": "user",
        "content": "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Shawn started with 5 toys."
    },
    {
        "role": "user",
        "content": "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer: Shawn started with 5 toys."
    },
    {
        "role": "assistant",
        "content": "If he got 2 toys each from his mom and dad, then that is 4 more toys."
    },
    {
        "role": "user",
        "content": "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys."
    },
    {
        "role": "assistant",
        "content": "5 + 4 = 9."
    },
    {
        "role": "user",
        "content": "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nAnswer: Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9."
    },
    {
        "role": "assistant",
        "content": "The answer is: 9"
    },
    {
        "role": "user",
        "content": "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "There were originally 9 computers."
    },
    {
        "role": "user",
        "content": "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer: There were originally 9 computers."
    },
    {
        "role": "assistant",
        "content": "For each of 4 days, 5 more computers were added."
    },
    {
        "role": "user",
        "content": "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer: There were originally 9 computers. For each of 4 days, 5 more computers were added."
    },
    {
        "role": "assistant",
        "content": "So 5 * 4 = 20 computers were added."
    },
    {
        "role": "user",
        "content": "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added."
    },
    {
        "role": "assistant",
        "content": "9 + 20 is 29."
    },
    {
        "role": "user",
        "content": "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nAnswer: There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29."
    },
    {
        "role": "assistant",
        "content": "The answer is: 29"
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Michael started with 58 golf balls."
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Michael started with 58 golf balls."
    },
    {
        "role": "assistant",
        "content": "After losing 23 on tuesday, he had 58 - 23 = 35."
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35."
    },
    {
        "role": "assistant",
        "content": "After losing 2 more, he had 35 - 2 = 33 golf balls."
    },
    {
        "role": "user",
        "content": "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday he lost 2 more. How many golf balls did he have at the end of wednesday?\nAnswer: Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls."
    },
    {
        "role": "assistant",
        "content": "The answer is: 33"
    },
    {
        "role": "user",
        "content": "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer:"
    },
    {
        "role": "assistant",
        "content": "Olivia had 23 dollars."
    },
    {
        "role": "user",
        "content": "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer: Olivia had 23 dollars."
    },
    {
        "role": "assistant",
        "content": "5 bagels for 3 dollars each will be 5 x 3 = 15 dollars."
    },
    {
        "role": "user",
        "content": "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars."
    },
    {
        "role": "assistant",
        "content": "So she has 23 - 15 dollars left."
    },
    {
        "role": "user",
        "content": "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left."
    },
    {
        "role": "assistant",
        "content": "23 - 15 is 8."
    },
    {
        "role": "user",
        "content": "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nAnswer: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8."
    },
    {
        "role": "assistant",
        "content": "The answer is: 8"
    }
]

# terminators = [
#     world_tokenizer.eos_token_id,
#     world_tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

ds = datasets.load_dataset("gsm8k", 'main')
data_list = list(ds['test'])
write_file = open("llama3_output_inference_single_step.txt", "w")
correct, total = 0, 0

# for each question in the test set of GSM8K
for i in range(len(data_list)):
    total += 1
    previous = "Question: " + data_list[i]['question'] + "\nAnswer: "
    answer = data_list[i]['answer'].split("####")[1].strip()
    for trail in range(15):
        new_messages = messages_gsm8k.copy()
        new_messages.append({
            "role": "user",
            "content": previous
        })
        conversations = tokenizer.apply_chat_template(
            new_messages,
            tokenize=False,
        )
        outputs = agent_model.generate(
            [conversations],
            SamplingParams(
                temperature=0.5,
                top_p=0.9,
                max_tokens=3000,
                stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  # KEYPOINT HERE
            )
        )
        
        response = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=True).replace("assistant\n\n", "")
        
        print("Previous: " + previous)
        print("Partial response: " + response)
        
        previous = previous + ' ' + response
        # world_input_ids = world_tokenizer.apply_chat_template(
        #     messages_start_rationale + messages_gsm8k + [
        #         {
        #             "role": "user",
        #             "content": previous
        #         }
        #     ],
        #     add_generation_prompt=True,
        #     return_tensors="pt"
        # ).to(world_model.device)
        # attention_mask = torch.ones_like(world_input_ids)
        # world_output = world_model.generate(
        #     world_input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=3000,
        #     eos_token_id=terminators,
        #     # temperature=0.1,
        #     return_dict_in_generate=True,
        #     output_scores=True,
        #     pad_token_id=world_tokenizer.eos_token_id
        # )
        # world_response = world_tokenizer.decode(world_output.sequences[0][world_input_ids.shape[1]:], skip_special_tokens=True)
        # transition_scores = world_model.compute_transition_scores(
        #     world_output.sequences, world_output.scores, normalize_logits=True
        # )
        # print("World response given partial response: " + world_response)
        # print("World response probability: " + str(np.exp(np.average(transition_scores.cpu().numpy()))))
        # print("-------------------------------------------------")
        if "The answer is:" in response:
            try:
                prediction = response.split("The answer is")[1].strip()
                if prediction == answer:
                    correct += 1
                print("final_answer: " + str(prediction) + '\t' + str(answer))
            except:
                print("Error")
            break
        if trail == 14:
            print("Stuck in loop")
            break

print("Accuracy: ", correct/total)
print("Total: ", total)
print("Correct: ", correct)