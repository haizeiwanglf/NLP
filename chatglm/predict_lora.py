from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("/data/wuguangshuo/THUDMchatglm-6b/", trust_remote_code=True)
model = AutoModel.from_pretrained("/data/wuguangshuo/THUDMchatglm-6b/", trust_remote_code=True)
model=model.half().cuda()
model = PeftModel.from_pretrained(model,"/home/wuguangshuo/dev_env/chatglm_tuning-main/output/adgen-chatglm-6b-lora/" )
model = model.eval()
#用来控制单轮或者多轮
single_round=True
history = []
while True:
    raw_input_text = input("Input:")
    if len(raw_input_text.strip()) == 0:
        break
    if single_round:
        sents = [raw_input_text]
        response = model.chat(sents)
    else:
        response, history = model.chat(raw_input_text, history=history)
    print("Response: ", response)
    print("\n")
