from openai import AsyncOpenAI
import random
import time
import asyncio
import httpx
# import nest_asyncio
from tqdm import tqdm
import argparse
import datasets
import json

# Those model cost numbers are current as of 2024-02-20
model_costs = {
    "gpt-3.5-turbo": {"context": 0.0015, "generated": 0.002},
    "gpt-3.5-turbo-0301": {"context": 0.0015, "generated": 0.002},
    "gpt-3.5-turbo-0613": {"context": 0.0015, "generated": 0.002},
    "gpt-3.5-turbo-16k": {"context": 0.003, "generated": 0.004},
    "gpt-3.5-turbo-16k-0613": {"context": 0.003, "generated": 0.004},
    "gpt-4": {"context": 0.03, "generated": 0.06},
    "gpt-4-32k": {"context": 0.06, "generated": 0.12},
    "gpt-4-0314": {"context": 0.03, "generated": 0.06},
    "gpt-4-0613": {"context": 0.03, "generated": 0.06},
    "gpt-4-32k": {"context": 0.06, "generated": 0.12},
    "gpt-4-32k-0314": {"context": 0.06, "generated": 0.12},
    "gpt-4-32k-0613": {"context": 0.06, "generated": 0.12},
    "gpt-4-0125-preview": {"context": 0.01, "generated": 0.03},
    "gpt-4-1106-preview": {"context": 0.01, "generated": 0.03},
    "gpt-4-1106-vision-preview": {"context": 0.01, "generated": 0.03},
    "gpt-4-turbo": {"context": 0.01, "generated": 0.03},
    "text-embedding-ada-002-v2": {"context": 0.0001, "generated": 0},
    "text-davinci:003": {"context": 0.03, "generated": 0.12},
    "whisper-1": {"context": 0.006 / 60, "generated": 0},
    "gpt-3.5-turbo-0125": {"context": 0.0005, "generated": 0.0015},
    "gpt-3.5-turbo-instruct": {"context": 0.0015, "generated": 0.002},
}

### BEGIN GENERAL FUNCTIONS ###

def get_price(model: str, response):
    # speciual case for fine-tuned model, assuming the base model is gpt-3.5-turbo 
    if model.startswith("ft"):
        price = response.usage.prompt_tokens / 1000 * 0.0030 + response.usage.completion_tokens / 1000 * 0.0060
    elif model in model_costs:
        price = response.usage.prompt_tokens / 1000 * model_costs[model]["context"] + response.usage.completion_tokens / 1000 * model_costs[model]["generated"]
    else:
        raise ValueError(f"Model {model} not found in model_costs")
    return price


async def api_call_single(client: AsyncOpenAI, model: str, messages: list[dict], pbar: tqdm, **kwargs):
    max_retries = 10  # Maximum number of retries
    retry_delay = 1.0  # Initial delay in seconds
    for attempt in range(max_retries):
        try:
            # Call the API
            response = await client.chat.completions.create(
                model=model,
                messages=messages,  # Ensure messages is a list
                **kwargs
            )
            # Calculate price based on model
            price = get_price(model, response)
            # Success, update progress bar and return response
            pbar.update(1) 
            return response, price
        
        except Exception as e:  # Replace Exception with your client's specific rate limit exception
            if isinstance(e, ValueError):
                # re-raising the exception if it's a ValueError because that came from get_price
                raise e
            if attempt < max_retries - 1:
                wait = retry_delay * (2 ** attempt)  # Exponential backoff formula
                print(f"Rate limit reached, retrying in {wait:.2f} seconds...")
                await asyncio.sleep(wait)
            else:
                print("Max retries reached, unable to complete request.")
                raise e  # Re-raise the last exception


def apply_async(client: AsyncOpenAI, model: str, messages_list: list[list[dict]], **kwargs):
    pbar = tqdm(total=len(messages_list), desc='Running API calls')
    loop = asyncio.get_event_loop()
    if loop.is_closed():
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = asyncio.get_event_loop()
    tasks = [loop.create_task(api_call_single(client, model, messages, pbar, **kwargs)) for messages in messages_list]
    result = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    total_price = sum([r[1] for r in result])
    response_list = [r[0] for r in result]
    return total_price, response_list

### END GENERAL FUNCTIONS ###

def get_messages_list(messages_list: str, system_prompt: str, example_pairs: list[dict]):
    # here we are only providing a dummy implementation so we're not actually reading the input file
    # replace this with your own implementation
    # the output format should be a list of lists of messages, where each message is a dictionary with the keys "role" and "content"
    messages = []
    for m in messages_list:
        current_message = [
            {
                "role": "system",
                "content": system_prompt,    
            },
        ]
        current_message.extend(example_pairs)
        current_message.append(
            {
                "role": "user",
                "content": m,
            }
        )
        messages.append(current_message)
    return messages


# template = """Here we have a question: "{q}"
# And the answer: "{a}"
# Your job is to add some thoughts in the necessary parts between words in the answer, which may be helpful for the answer generation.
# The thoughts should begin with <BOT> and end with <EOT>
# Answer with thoughts:
# """

system_prompt = """Your task is to add rationals to a piece of text. The rationals should help you with predicting future text. You can add rationals by writing <BOT>rational<EOT>"""

example_pairs = [
    {
        "role": "user",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72",
    },
    {
        "role": "assistant",
        "content": "Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\nAnswer: Natalia sold 48/2 <BOT>48 / 2 should be 24<EOT>= <<48/2=24>>24 clips in May. <BOT>We have already calculated the number of chips in May, now we should calculate the sum of chips in April and May<EOT>Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72",
    },
    {
        "role": "user",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
    },
    {
        "role": "assistant",
        "content": "Quesetion: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?\nAnswer: <BOT>To find out how much Weng earns per minute, we need to divide her hourly rate by the number of minutes in an hour<EOT> Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.<BOT>Now, to find out her total earnings for 50 minutes, we multiply the rate per minute by the number of minutes she worked<EOT> Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10"
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="input.txt", required=False)
    parser.add_argument('--output_path', type=str, default="results.csv", required=False)
    # parser.add_argument("--model", type=str, default="gpt-4-0125-preview", help="The model to use for generation", required=False)
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-0125", help="The model to use for generation", required=False)
    args = parser.parse_args()

    ds = datasets.load_dataset("gsm8k", 'main')
    client = AsyncOpenAI(
        http_client=httpx.AsyncClient(
            limits=httpx.Limits(
            max_connections=100000,
            max_keepalive_connections=100
            )
        )
    )
    f = open(args.output_path, "w")
    
    messages_list = []
    data_list = list(ds['train'])[0: 1000]
    for i in range(len(data_list)):
        messages_list.append("Question: " + data_list[i]['question'] + "\nAnswer: " + data_list[i]['answer'])
    
    # get messages list to be used as input
    messages_list = get_messages_list(system_prompt=system_prompt, example_pairs=example_pairs, messages_list=messages_list)
    # TODO: give your own keyword arguments here
    kwargs = {
        "temperature": 0,
    }
    # do a test call on one instance to estimate the total cost
    random_number = random.randint(0, len(messages_list) - 1)
    print("Testing on message number " + str(random_number) + " to estimate cost...")
    random_response_list = apply_async(client, args.model, [messages_list[random_number]], **kwargs)
    price = random_response_list[0] * len(messages_list)
    print("You are running: " + args.model + ". Estimated cost: $" + str(round(price, 3)))
    # print(random_response_list)
    if price > 100 and price <= 1000:
        input("Estimated cost is above $100, are you sure you want to proceed? Press enter to continue.")
        # The script will continue after the user presses Enter.
    elif price > 1000:
        raise ValueError("Estimated cost is above $1000, are you trying to bankrupt Daniel? Please contact him first before proceeding!")
    leftover_message_list = messages_list[0: random_number] + messages_list[random_number + 1:]
    print("Running leftover messages...")
    response_list = apply_async(client, args.model, leftover_message_list, **kwargs)
    print("Actual cost: " + str(round(response_list[0], 3)) + "$")
    final_answer_list = response_list[1][0: random_number] + random_response_list[1] + response_list[1][random_number:]
    # do post-processing here, create a json file with all the input and output data
    for i, response in enumerate(final_answer_list):
        d = {
            "input": messages_list[i],
            "output": response.choices[0].message.content
        }
        f.write(json.dumps(d) + "\n")
print(f"Output written to {args.output_path}")
