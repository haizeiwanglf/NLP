import sentencepiece as spm

# makes segmenter instance and loads the model file (m.model)
sp = spm.SentencePieceProcessor()
sp.load('./spm_model/gpt.model')

text = """
垃圾分类，一般是指按一定规定或标准将垃圾分类储存、投放和搬运，从而转变成公共资源的一系列活动的总称。
"""
# encode: text => id
print(sp.encode_as_pieces(text))
print(sp.encode_as_ids(text))
print(len(sp.encode_as_pieces(text)))
print(len(sp.encode_as_ids(text)))
