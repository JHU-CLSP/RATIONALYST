import json

correct, total = 0, 0
write_file = open("llm_training_data.jsonl", "w")

for line in open("../llama3_output_with_score.txt"):
    if not line.startswith("-----"):
        total += 1
        d = json.loads(line.strip())
        
        # split string at the position of <BOT>
        splits = d['output'].split(" <BOT>")
        real_output = splits[0]
        for split in splits[1:]:
            try:
                leftover = split.split("<EOT>")[1]
                real_output += leftover
            except:
                continue
        input = d['input'].replace("\n", " ")
        
        # we only keep the generated output if without rationales it's the same as the input
        # if d['input'].replace("\n", " ").replace(" ", '') == real_output.replace("\n", " ").replace("  ", " ").replace(' ', ''):
        if d['input'].replace("\n", " ").replace(" ", '') == real_output.replace("\n", " ").replace(" ", ''):
            correct += 1
            # find the pairs of preceeding text and rationale
            splits = d['output'].split("<BOT>")
            preceeding = ""
            for i in range(1, len(splits)):
                try:
                    # get the preceeding text
                    preceeding += splits[i-1].split("<EOT>")[1]
                except:
                    preceeding += splits[i-1]
                # get the rationale
                rationale = splits[i].split("<EOT>")[0]
                # sometimes we can't find end of thought tokens
                try:
                    following = splits[i].split("<EOT>")[1]
                except:
                    continue
                # print("proceeding: " + preceeding)
                # print("rationale: " + rationale)
                # print("following: " + following)
                # print("----------------------------------------\n")
                write_file.write(json.dumps({"preceeding": preceeding, "rationale": rationale, "following": following}) + "\n")
        
print(correct, total, correct / total)
