# NLP核心概念

*精选的自然语言处理核心概念和代码*

---

## 1. 文本预处理

### 1.1 Tokenization

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 分词
text = "Hello, world!"
tokens = tokenizer.tokenize(text)
# ['Hello', ',', 'world', '!']

# 编码
encoding = tokenizer(text, return_tensors='pt')
# {'input_ids': tensor([[ 101, 7592, 1010, 2088, 1029,  102]]),
#  'token_type_ids': tensor([[0, 0, 0, 0, 0, 0]]),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1]])}

# 批量编码
texts = ["Hello world", "How are you?"]
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
```

### 1.2 特殊Token

- `[CLS]`：分类token，位于句首
- `[SEP]`：分隔token，句子结束
- `[PAD]`：填充token，补齐长度
- `[UNK]`：未知词

---

## 2. 语言模型

### 2.1 BERT

```python
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inputs = tokenizer("Hello, world!", return_tensors='pt')
outputs = model(**inputs)

# last_hidden_state: [batch, seq_len, hidden_dim]
# pooler_output: [batch, hidden_dim]
pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
```

### 2.2 GPT

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_text = "Once upon a time"
inputs = tokenizer(input_text, return_tensors='pt')

# 生成文本
outputs = model.generate(**inputs, max_length=100, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### 2.3 RoBERTa

```python
from transformers import RobertaModel, RobertaTokenizer

model = RobertaModel.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
```

---

## 3. 文本分类

### 3.1 BERT分类器

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

# 训练
model = BertClassifier(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    for input_ids, attention_mask, labels in train_loader:
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 3.2 情感分析

```python
from transformers import pipeline

# 使用预训练情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# 中文情感分析
classifier = pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-chinanews-chinese")
result = classifier("这个产品很好用")
```

---

## 4. 命名实体识别（NER）

```python
from transformers import pipeline

# 实体识别
ner = pipeline("ner", aggregation_strategy="simple")
result = ner("John lives in New York")
# [{'entity_group': 'PER', 'word': 'John', 'score': 0.99},
#  {'entity_group': 'LOC', 'word': 'New York', 'score': 0.98}]

# 使用BERT进行NER
from transformers import BertForTokenClassification, BertTokenizer

model = BertForTokenClassification.from_pretrained('dslim/bert-base-NER')
tokenizer = BertTokenizer.from_pretrained('dslim/bert-base-NER')

inputs = tokenizer("John lives in New York", return_tensors='pt', truncation=True)
outputs = model(**inputs).logits
predictions = torch.argmax(outputs, dim=2)
```

---

## 5. 问答系统

### 5.1 抽取式问答

```python
from transformers import pipeline

# 问答
qa = pipeline("question-answering")
result = qa(question="What is the capital of France?", 
            context="Paris is the capital of France.")
# {'answer': 'Paris', 'score': 0.99, 'start': 0, 'end': 5}
```

### 5.2 BERT问答

```python
from transformers import BertForQuestionAnswering, BertTokenizer

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

question = "What is the capital of France?"
context = "Paris is the capital of France."

inputs = tokenizer(question, context, return_tensors='pt')
outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

start_index = torch.argmax(start_scores)
end_index = torch.argmax(end_scores)
answer = tokenizer.decode(inputs['input_ids'][0][start_index:end_index+1])
```

---

## 6. 文本生成

### 6.1 GPT生成

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "Once upon a time"
inputs = tokenizer(prompt, return_tensors='pt')

# 贪婪搜索
outputs = model.generate(inputs['input_ids'], max_length=100)
print(tokenizer.decode(outputs[0]))

# 束搜索
outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=5)
print(tokenizer.decode(outputs[0]))

# Nucleus采样
outputs = model.generate(inputs['input_ids'], max_length=100, 
                        do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.decode(outputs[0]))
```

### 6.2 文本摘要

```python
from transformers import pipeline

# 摘要
summarizer = pipeline("summarization")
result = summarizer(article, max_length=130, min_length=30)
```

---

## 7. 机器翻译

```python
from transformers import pipeline

# 翻译
translator = pipeline("translation_en_to_fr")
result = translator("Hello, how are you?")
# [{'translation_text': 'Bonjour, comment allez-vous?'}]

# 使用MarianMT
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-zh"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

inputs = tokenizer("Hello, world!", return_tensors="pt")
translated = model.generate(**inputs)
result = tokenizer.decode(translated[0], skip_special_tokens=True)
```

---

## 8. 文本相似度

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 编码句子
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["I love cats", "I like animals", "The weather is nice"]
embeddings = model.encode(sentences)

# 计算相似度
similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
print(f"Similarity: {similarity[0][0]}")
```

---

## 9. 提示工程

### 9.1 Few-shot提示

```python
prompt = """
Classify the sentiment of these reviews:

Review: "This product is amazing!"
Sentiment: Positive

Review: "This product is terrible."
Sentiment: Negative

Review: "It's okay, not great."
Sentiment:
"""
```

### 9.2 Chain-of-Thought提示

```python
prompt = """
Solve this problem step by step:

If I have 5 apples and I buy 3 more apples, then I eat 2 apples, 
how many apples do I have?

Let's think step by step:
1. Starting with 5 apples
2. Buying 3 more: 5 + 3 = 8 apples
3. Eating 2: 8 - 2 = 6 apples

Answer: 6 apples
```

---

## 10. 微调技术

### 10.1 LoRA微调

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query", "value"],
    lora_dropout=0.05,
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 884736 || all params: 124615808 || trainable%: 0.71
```

### 10.2 量化微调

```python
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=quantization_config,
)
```

---

*NLP核心概念整理完成！* 📚💬
