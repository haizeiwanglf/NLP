from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/data/wuguangshuo/THUDMchatglm-6b/", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/wuguangshuo/THUDMchatglm-6b/", trust_remote_code=True)
model=model.half().cuda()
model = PeftModel.from_pretrained(model,"/home/wuguangshuo/dev_env/chatglm_tuning-main/output/adgen-chatglm-6b-lora/" )
model = model.eval()
door=True
while door:
    query = input("\n用户: ")
    # query = input("")
    if query=="\n用户: exit":
        break
    response, history = model.chat(tokenizer, query, history=[])
    print(response)