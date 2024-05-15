import asyncio
import aiohttp
from tqdm import tqdm
import random
import datasets
import json
import time

base_url = ['http://c013', 'http://c007', 'http://c009']
ports = [1233, 1234, 1235, 1236, 1237, 1238, 1239, 1240]

def get_url():
    return random.choice(base_url) + ':' + str(random.choice(ports)) + '/v1/chat/completions'

messages_start_rationale = [
    {
        "role": "system",
        "content": "Your task is to add rationales given to a piece of text. The rationales should be added after each sentence. The rationales should help you with predicting future text. You can add rationales by writing <BOT>rational<EOT>. Other than the rationales, please do not modify the original text."
    },
]

# Load dataset in streaming mode
ds = datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True)

async def safe_request(session, method, url, headers, json, retries=3):
    for attempt in range(retries):
        try:
            async with session.request(method, url, headers=headers, json=json) as response:
                response.raise_for_status()  # Will raise an error for 4XX/5XX responses
                return await response.json()
        except aiohttp.ClientError as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff
            else:
                raise

async def send_request(session, data):
    try:
        new_messages = messages_start_rationale.copy()
        new_messages.append({
            "role": "user",
            "content": data
        })
        url = get_url()
        content = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": new_messages,
            "max_tokens": 2000,
            "temperature": 0.0,
            "stop_token_ids": [128001, 128009],
            "seed": 14
        }
        headers = {"Content-Type": "application/json"}
        return await safe_request(session, 'POST', url, headers, content)
    except Exception as e:
        print(f"Failed to send request: {e}")
        return {"input": data, "output": "Error"}


async def main(output_file):
    async with aiohttp.ClientSession() as session:
        pbar = tqdm(ds, desc="Processing texts", unit="text")
        tasks = []
        responses = []
        for data in pbar:
            text = data['text'][0: 100]
            if text.strip():  # ensure not to send empty text
                task = asyncio.create_task(send_request(session, text))
                tasks.append(task)
                if len(tasks) >= 2000:  # Manage request concurrency
                    batch_responses = await asyncio.gather(*tasks)
                    responses.extend(batch_responses)
                    tasks = []
                    for response in batch_responses:
                        output_file.write(json.dumps({"input": text, "output": response}) + '\n')

        if tasks:  # Catch any remaining tasks
            batch_responses = await asyncio.gather(*tasks)
            responses.extend(batch_responses)
            for response in batch_responses:
                output_file.write(json.dumps({"input": text, "output": response}) + '\n')

        return responses

if __name__ == '__main__':
    start_time = time.time()
    with open("output_with_responses.txt", "w") as output_file:
        asyncio.run(main(output_file))
    print("Total time: ", time.time() - start_time)
