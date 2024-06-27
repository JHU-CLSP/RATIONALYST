import json
import sys

cot_lines = open(sys.argv[1]).readlines()
total, correct, has_answer = 0, 0, 0

for i in range(len(cot_lines)):
    total += 1
    d_cot = json.loads(cot_lines[i])
    # if d_cot["normalized_answer"] == d_cot["normalized_prediction"]:
    if d_cot["answer"] == d_cot["normalized_prediction"] or d_cot["normalized_answer"] == d_cot["normalized_prediction"] and d_cot["normalized_answer"] != '':
        correct += 1
    # else:
    #     print(d_cot["full_response"])
    if  d_cot["normalized_prediction"] != "":
        has_answer += 1
        
print('Correct: ' + str(correct) + ' Has answer: ' + str(has_answer) + ' Total: ' + str(total) + ' Real Total: ' + str(1319))
print(correct / total * 100)

