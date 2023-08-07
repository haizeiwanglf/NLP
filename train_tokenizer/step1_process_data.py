import os
import json

corpus=open('data/corpus.txt','w',encoding='utf-8')
cnt=0

root_path="/data/wuguangshuo/dataset/wiki_zh_2019/wiki_zh"
path_1_list=os.listdir(root_path)
for path_1 in path_1_list:
    temp=os.path.join(root_path,path_1)
    path_2_list=os.listdir(temp)
    for path_2 in path_2_list:
        with open(os.path.join(temp,path_2),"r",encoding="utf-8") as f:
            for line in f.readlines():
                data=json.loads(line)
                text=data["text"]
                corpus.write(text.strip()+'\n')
                cnt+=1
                # if cnt>5000:
                #     break
print(cnt)
corpus.close()