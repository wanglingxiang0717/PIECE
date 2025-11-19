from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# 分类任务示例 (MRPC)
batch_class = [
    {"sentence1": "The cat sat on the mat.", "sentence2": "A cat is on a rug.", "label": 1},
    {"sentence1": "Dogs are friendly.", "sentence2": "Cats are unfriendly.", "label": 0},
]

collator_class = DataCollatorForGLUE(tokenizer=tokenizer, max_length=128, task_type="classification")
inputs_class = collator_class(batch_class)
print(inputs_class)

# 回归任务示例 (STS-B)
batch_reg = [
    {"sentence1": "A man is playing guitar.", "sentence2": "A person plays a musical instrument.", "label": 4.5},
    {"sentence1": "Kids are running outside.", "sentence2": "Children run outdoors.", "label": 5.0},
]

collator_reg = DataCollatorForGLUE(tokenizer=tokenizer, max_length=128, task_type="regression")
inputs_reg = collator_reg(batch_reg)
print(inputs_reg)
