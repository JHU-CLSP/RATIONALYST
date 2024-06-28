import json
import numpy as np
import random
import sys

gsm8k_gt = {}
exponential = 0.9

'''
for line in open('gsm8k_gt_annotation.tsv').readlines()[1:]:
    line = line.strip()
    if line == '':
        continue
    line = line.split('\t')
    # content of the line include id, preceeding, rationale, following, CY annotation, DW annotation, Diff
    # key = 'preceeding: ' + line[1].strip() + ' rationale: ' + line[2].strip() + ' following: ' + line[3].strip()
    key = 'preceeding: ' + line[1].strip() + ' following: ' + line[3].strip()
    key = key.replace('\n', '').replace(" ", '')
    # 1 for ratioinale useful for following, 0 for ratioinale not useful for following
    gsm8k_gt[key] = int(line[5])

A, B, C, D = 0, 0, 0, 0
total, filtered = 0, 0

for line in open("llm_training_data_filtered_new_without_final.jsonl"):
    d = json.loads(line)
    # key = 'preceeding: ' + d['preceeding'].strip() + ' rationale: ' + d['rationale'].strip() + ' following: ' + d['following'].strip()
    key = 'preceeding: ' + d['preceeding'].strip() + ' following: ' + d['following'].strip()
    key = key.replace('\n', '').replace(" ", '')
    # calculate percision and recall
    # 1 for ratioinale useful for following, 0 for ratioinale not useful for following
    perplexity_without_rationale_list = d['perplexity_without_rationale_list']
    perplexity_with_rationale_list = d['perplexity_with_rationale_list']
    # is_correct = sum(perplexity_without_rationale_list) > sum(perplexity_with_rationale_list)
    # is_correct = int(d['is_correct'])
    # perplexity_without_rationale = d['perplexity_without_rationale']
    # perplexity_with_rationale = d['perplexity_with_rationale']
    perplexity_with_rationale, perplexity_without_rationale = 0, 0
    for i in range(0, len(perplexity_with_rationale_list)):
        perplexity_with_rationale += perplexity_with_rationale_list[i] * np.power(exponential, i)
    for i in range(0, len(perplexity_without_rationale_list)):
        perplexity_without_rationale += perplexity_without_rationale_list[i] * np.power(exponential, i)
    
    # print(len(perplexity_with_rationale_list), len(perplexity_without_rationale_list))
    # print(perplexity_with_rationale, perplexity_without_rationale)
    is_correct = perplexity_without_rationale < perplexity_with_rationale
    total += 1
    if is_correct == 1:
        filtered += 1
        
    if key not in gsm8k_gt:
        continue
    
    # print(is_correct, gsm8k_gt[key])
    if is_correct == 1 and gsm8k_gt[key] == 1:
        A += 1
    elif is_correct == 1 and gsm8k_gt[key] == 0:
        B += 1
        print(d["preceeding"])
        print(d["rationale"])
        print(d["following"])
    elif is_correct == 0 and gsm8k_gt[key] == 1:
        C += 1
    elif is_correct == 0 and gsm8k_gt[key] == 0:
        D += 1
    

print(A, B, C, D)
try:
    precision = A / (A + B)
    recall = A / (A + C)
    f1 = 2 * precision * recall / (precision + recall)
except ZeroDivisionError:
    precision = 0
    recall = 0
    f1 = 0
print("Total: ", A + B + C + D)
print(f'precision: {precision}, recall: {recall}, f1: {f1}')
print(f'filtered: {filtered}, total: {total}', filtered / total)
'''

write_file_train, write_file_dev = open('../sampling_results/all_datasets_train.jsonl', 'w'), open('../sampling_results/all_datasets_dev.jsonl', 'w')
# split the data into training and dev, don't do any more operation, just split 95% to training and 5% to dev
lines = open(sys.argv[1]).readlines()
random.shuffle(lines)
temp_lines = []
for line in lines:
    d = json.loads(line)
    if d['is_correct'] == 1:
        temp_lines.append(line)
lines = temp_lines

for i in range(0, int(len(lines) * 0.95)):
    write_file_train.write(lines[i])
for i in range(int(len(lines) * 0.95), len(lines)):
    write_file_dev.write(lines[i])

