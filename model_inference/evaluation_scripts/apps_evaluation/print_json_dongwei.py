import json
import sys



write_file = open("output.txt", 'w')
# for line in open("output_dongwei.json"):
for line in open(sys.argv[1]):
    write_file.write(json.loads(line.strip())['full_response'])
    write_file.write('\n-------------------\n')
