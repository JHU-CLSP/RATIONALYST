import re
import string
from nltk.corpus import stopwords
from datasets import load_dataset
import numpy as np
import time
import sys
import gc
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import psutil
from InstructorEmbedding import INSTRUCTOR

instruction = "Represent the reasoning type for retrieval:"


worker_id = int(sys.argv[1])  # Ensure this is an integer
# worker_id = 0
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda:" + str(worker_id))

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 ** 2)} MB")


start_time = time.time()
selected = ['FreeLaw', 'StackExchange', 'Wikipedia (en)', 'DM Mathematics', 'Ubuntu IRC', 'HackerNews']

# Load and preprocess your documents
gsm8k_data_list = load_dataset("gsm8k", 'main')
gsm8k_data_list = list(gsm8k_data_list['train'])
# gsm8k_documents = ["Question: " + gsm8k_data['question'] + "\nAnswer: " + gsm8k_data['answer'] for gsm8k_data in gsm8k_data_list]
gsm8k_documents = [gsm8k_data['question'] + "\n" + gsm8k_data['answer'] for gsm8k_data in gsm8k_data_list]

ecqa_data_list = load_dataset("yangdong/ecqa")['train']
# ecqa_documents = ["Question: " + ecqa_data['q_text'] + "\nAnswer Choices: " + " ".join([ecqa_data[f'q_op{i}'] for i in range(1, 5)]) for ecqa_data in ecqa_data_list]
ecqa_documents = [ecqa_data['q_text'] + "\n" + " ".join([ecqa_data[f'q_op{i}'] for i in range(1, 5)]) for ecqa_data in ecqa_data_list]


arc_data_list = load_dataset("ai2_arc", 'ARC-Challenge')['train']
# arc_documents = ["Question: " + arc_data['question'] + "\nAnswer Choices: " + " ".join(arc_data['choices']['text']) for arc_data in arc_data_list]
arc_documents = [arc_data['question'] + "\n" + " ".join(arc_data['choices']['text']) for arc_data in arc_data_list]


proofwriter_data_list = load_dataset("tasksource/proofwriter")['train']
# proofwriter_documents = ["Question: " + proofwriter_data['question'] + "\nTheory: " + proofwriter_data['theory'] for proofwriter_data in proofwriter_data_list]
proofwriter_documents = [proofwriter_data['question'] + "\n" + proofwriter_data['theory'] for proofwriter_data in proofwriter_data_list]

your_documents = gsm8k_documents + ecqa_documents + arc_documents + proofwriter_documents

# Preprocess function
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

preprocessed_documents = [preprocess(doc) for doc in your_documents]

# Load the embedding model
# embeddings = model.encode([[instruction, sentence]])

# Generate embeddings for each document and compute the average embedding
document_embeddings = model.encode([preprocessed_documents[0]])
document_embeddings = model.encode(preprocessed_documents)
your_embeddings = [np.mean(document_embeddings, axis=0)]

# Load Pile documents in chunks
total_pile_documents = load_dataset("monology/pile-uncopyrighted", split="train")
length = len(total_pile_documents)

def load_pile_documents_in_chunks(dataset, chunk_size, start_index, end_index):
    for i in range(start_index, end_index, chunk_size):
        yield dataset[i:i+chunk_size]

write_file = open("filtered_pile_documents" + str(worker_id) + ".txt", "w")

chunk_size = 30000  # Smaller chunk size

for pile_documents in load_pile_documents_in_chunks(total_pile_documents, chunk_size, worker_id * length // 8, (worker_id + 1) * length // 8):
    new_pile_documents = []
    
    for j in range(len(pile_documents['text'])):
        # if pile_documents['meta'][j]["pile_set_name"] not in selected or len(pile_documents['text'][j]) > 2000:
        #     continue
        if len(pile_documents['text'][j]) > 2000:
             continue
        new_pile_documents.append((pile_documents["text"][j], pile_documents['meta'][j]["pile_set_name"]))
    
    pile_documents_text = [new_pile_document[0] for new_pile_document in new_pile_documents]
    print(len(pile_documents_text))
    
    pile_preprocessed = [preprocess(doc) for doc in pile_documents_text]
    pile_embeddings = model.encode(pile_preprocessed)
    
    similarity_matrix = cosine_similarity(your_embeddings, pile_embeddings)
    similarity_matrix = similarity_matrix.reshape(-1)
    
    for j, similarity in enumerate(similarity_matrix):
        if similarity >= 0.3:
            print(new_pile_documents[j][1])
            write_file.write(pile_documents_text[j] + "\n---------------------------------------------\n")
    
    # Free memory
    del pile_preprocessed, pile_embeddings, similarity_matrix, pile_documents_text, new_pile_documents
    gc.collect()  # Force garbage collection
    print_memory_usage()  # Monitor memory usage

    print("--- %s seconds ---" % (time.time() - start_time))
    print("chunk done")

write_file.close()
