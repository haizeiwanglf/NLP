from pruners.vocabulary_pruner import BloomVocabularyPruner

# 需要进行裁剪的模型路径
# model_name_or_path = '/data/wuguangshuo/llm_models/bigsciencebloom-560m'
# # 自己制作的词表的路
# new_tokenizer_name_or_path = '/data/wuguangshuo/llm_models/YeungNLPbloom-396m-zh'
model_name_or_path = 'E:\llm_models\/bigsciencebloom-560m'
# 自己制作的词表的路
new_tokenizer_name_or_path = 'E:\llm_models\YeungNLPbloom-396m-zh'
save_path = 'path-to-save'
pruner = BloomVocabularyPruner()
# 裁剪
pruner.prune(model_name_or_path, new_tokenizer_name_or_path, save_path)
# 检查裁剪的模型与原模型是否一致
pruner.check(model_name_or_path, save_path, text='长风破浪会有时')