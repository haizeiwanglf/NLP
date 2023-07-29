

import json

wr = open("./train_huatuo.json", mode="w",encoding="utf-8")
num=0
with open("./train_datasets.jsonl", mode="r", encoding="utf-8") as f:
    for line in f:
        if num>30000:
            break
        data = json.loads(line.strip())
        json_dict = {}
        json_dict["instruction"] = data["questions"][0]
        json_dict["output"] = data["answers"][0]
        wr.write(json.dumps(json_dict, ensure_ascii=False))
        wr.write("\n")
        num+=1
print(num)#796444
f.close()

