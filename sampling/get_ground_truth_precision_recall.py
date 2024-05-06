import json

gsm8k_gt = {}

for line in open('gsm8k_gt_annotation.tsv').readlines()[1:]:
    line = line.strip()
    if line == '':
        continue
    line = line.split('\t')
    # content of the line include id, preceeding, rationale, following, CY annotation, DW annotation, Diff
    key = 'preceeding: ' + line[1].strip() + ' rationale: ' + line[2].strip() + ' following: ' + line[3].strip()
    key = key.replace('\n', '').replace(" ", '')
    # 1 for ratioinale useful for following, 0 for ratioinale not useful for following
    gsm8k_gt[key] = int(line[5])

A, B, C, D = 0, 0, 0, 0
total, filtered = 0, 0

for line in open("llm_training_data_filtered.jsonl"):
    d = json.loads(line)
    key = 'preceeding: ' + d['preceeding'].strip() + ' rationale: ' + d['rationale'].strip() + ' following: ' + d['following'].strip()
    key = key.replace('\n', '').replace(" ", '')
    # calculate percision and recall
    # 1 for ratioinale useful for following, 0 for ratioinale not useful for following
    perplexity_without_rationale_list = d['perplexity_without_rationale_list']
    perplexity_with_rationale_list = d['perplexity_with_rationale_list']
    # is_correct = sum(perplexity_without_rationale_list) > sum(perplexity_with_rationale_list)
    is_correct = int(d['is_correct'])
    total += 1
    if is_correct == 1:
        filtered += 1
        
    if key not in gsm8k_gt:
        continue
    
    print(is_correct, gsm8k_gt[key])
    if is_correct == 1 and gsm8k_gt[key] == 1:
        A += 1
    elif is_correct == 1 and gsm8k_gt[key] == 0:
        B += 1
    elif is_correct == 0 and gsm8k_gt[key] == 1:
        C += 1
    elif is_correct == 0 and gsm8k_gt[key] == 0:
        D += 1
    

print(A, B, C, D)
precision = A / (A + B)
recall = A / (A + C)
f1 = 2 * precision * recall / (precision + recall)
print("Total: ", A + B + C + D)
print(f'precision: {precision}, recall: {recall}, f1: {f1}')
print(f'filtered: {filtered}, total: {total}', filtered / total)
