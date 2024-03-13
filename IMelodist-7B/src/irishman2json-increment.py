import os
import json

data = []
with open("datasets/v1_0309_yifan/irishman.json", 'r') as f:
    for line in f:
        data.append(json.loads(line))

conversations = []
for d in data:
    output = d["abc notation"]
    if len(output) <= 2900:
        continue
    conversation = {
            "system": "",
            "input": "",
            "output": output
        }
    conversations.append({"conversation": [conversation]})

filenames = os.listdir(dir)

for filename in filenames:
    filepath = os.path.join(dir,filename)
    with open(filepath,'r') as f:
        js = json.load(f)
    print(f"{filename} data number:",len(js))
    conversations.extend(js)

with open("./datasets/v1_0309_yifan/v1-all-in.json",'w') as f:
    json.dump(conversations,f,indent=4)