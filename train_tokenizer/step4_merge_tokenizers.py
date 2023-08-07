#!/usr/bin/env python
# -*- coding:utf-8 _*-
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]="python"
from transformers import LlamaTokenizer
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm
import argparse

model_file= 'merge.model' #并后的spm tokenizer
vocab_size='40k'#合并后的llama tokenizer

llama_tokenizer_dir = "/data/wuguangshuo/llm_models/llama2-hf/tokenizer.model"
chinese_sp_model_file = "./spm_model/gpt.model"

# load
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)# 原生LLaMA分词模型
chinese_sp_model = spm.SentencePieceProcessor()
chinese_sp_model.Load(chinese_sp_model_file)

llama_spm = sp_pb2_model.ModelProto()
llama_spm.ParseFromString(llama_tokenizer.sp_model.serialized_model_proto())
chinese_spm = sp_pb2_model.ModelProto()
chinese_spm.ParseFromString(chinese_sp_model.serialized_model_proto())

# print number of tokens
print(len(llama_tokenizer),len(chinese_sp_model))
print(llama_tokenizer.all_special_tokens)
print(llama_tokenizer.all_special_ids)
print(llama_tokenizer.special_tokens_map)

## Add Chinese tokens to LLaMA tokenizer
llama_spm_tokens_set=set(p.piece for p in llama_spm.pieces)
print(len(llama_spm_tokens_set))
print(f"Before:{len(llama_spm_tokens_set)}")
for p in chinese_spm.pieces:
    piece = p.piece
    if piece not in llama_spm_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        new_p.score = 0
        llama_spm.pieces.append(new_p) # 将训练的分词模型追加新的token到之前的模型
print(f"New model pieces: {len(llama_spm.pieces)}")

## Save
output_sp_dir = 'merged_tokenizer_sp'
output_hf_dir = f'merged_tokenizer_hf_{vocab_size}' # the path to save Chinese-LLaMA tokenizer
os.makedirs(output_sp_dir,exist_ok=True)
with open(output_sp_dir+f'/{model_file}', 'wb') as f:
    f.write(llama_spm.SerializeToString())
tokenizer = LlamaTokenizer(vocab_file=output_sp_dir+f'/{model_file}')

tokenizer.save_pretrained(output_hf_dir)
print(f"Chinese-LLaMA tokenizer has been saved to {output_hf_dir}")


# Test
llama_tokenizer = LlamaTokenizer.from_pretrained(llama_tokenizer_dir)
chinese_llama_tokenizer = LlamaTokenizer.from_pretrained(output_hf_dir)
print(tokenizer.all_special_tokens)
print(tokenizer.all_special_ids)
print(tokenizer.special_tokens_map)
text='''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。'''
#text='''大模型是指具有非常大的参数数量的人工神经网络模型。 在深度学习领域，大模型通常是指具有数亿到数万亿参数的模型。'''
print("Test text:\n",text)
print(f"Tokenized by LLaMA tokenizer:{len(llama_tokenizer.tokenize(text))},{llama_tokenizer.tokenize(text)}")
print(f"Tokenized by GoGPT-LLaMA tokenizer:{len(chinese_llama_tokenizer.tokenize(text))},{chinese_llama_tokenizer.tokenize(text)}")
