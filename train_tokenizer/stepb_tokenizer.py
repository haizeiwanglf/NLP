from transformers import BloomTokenizerFast, BloomForCausalLM
tokenizer = BloomTokenizerFast.from_pretrained('/home/wuguangshuo/dev_env/train_tokenizer/path-to-save/')
model = BloomForCausalLM.from_pretrained('/home/wuguangshuo/dev_env/train_tokenizer/path-to-save')
print(tokenizer.batch_decode(model.generate(tokenizer.encode('长风破浪会有时', return_tensors='pt'))))