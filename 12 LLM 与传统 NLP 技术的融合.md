
# 12 LLM 与传统 NLP 技术的融合

## 12.1 LLM 增强信息检索

### 12.1.1 语义搜索实现

语义搜索利用LLM的语义理解能力，提高搜索结果的相关性和准确性。

```python
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def encode(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        return model_output.last_hidden_state.mean(dim=1).numpy()

    def search(self, query, documents, top_k=5):
        query_embedding = self.encode([query])
        doc_embeddings = self.encode(documents)
        
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        return [(documents[i], similarities[i]) for i in top_indices]

# 使用示例
semantic_search = SemanticSearch()
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "I think, therefore I am"
]
query = "What is the meaning of life?"

results = semantic_search.search(query, documents)
for doc, score in results:
    print(f"Score: {score:.4f}, Document: {doc}")
```

### 12.1.2 问答系统优化

结合LLM和传统检索技术可以构建更强大的问答系统。

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

class EnhancedQASystem:
    def __init__(self, model_name='deepset/roberta-base-squad2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        self.semantic_search = SemanticSearch()  # 使用之前定义的SemanticSearch类

    def answer_question(self, question, context):
        inputs = self.tokenizer(question, context, return_tensors='pt')
        outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        
        answer = self.tokenizer.decode(inputs.input_ids[0][answer_start:answer_end])
        return answer

    def retrieve_and_answer(self, question, documents, top_k=3):
        relevant_docs = self.semantic_search.search(question, documents, top_k)
        
        best_answer = ""
        best_score = -float('inf')
        
        for doc, score in relevant_docs:
            answer = self.answer_question(question, doc)
            if score > best_score:
                best_score = score
                best_answer = answer
        
        return best_answer

# 使用示例
qa_system = EnhancedQASystem()
documents = [
    "The Earth is the third planet from the Sun and the only astronomical object known to harbor life.",
    "Water is a transparent, tasteless, odorless, and nearly colorless chemical substance.",
    "Photosynthesis is the process by which plants use sunlight, water and carbon dioxide to produce oxygen and energy in the form of sugar."
]
question = "What is the process by which plants produce oxygen?"

answer = qa_system.retrieve_and_answer(question, documents)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

### 12.1.3 文档摘要生成

LLM可以用于生成高质量的文档摘要，结合传统的提取式摘要方法可以进一步提高效果。

```python
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class HybridSummarizer:
    def __init__(self):
        self.abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def extractive_summarize(self, text, num_sentences=3):
        sentences = text.split('.')
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
        sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
        top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]
        extractive_summary = '. '.join([sentences[i].strip() for i in sorted(top_sentence_indices)])
        return extractive_summary

    def abstractive_summarize(self, text, max_length=150):
        summary = self.abstractive_summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    def hybrid_summarize(self, text, extractive_sentences=3, max_length=150):
        extractive_summary = self.extractive_summarize(text, extractive_sentences)
        final_summary = self.abstractive_summarize(extractive_summary, max_length)
        return final_summary

# 使用示例
summarizer = HybridSummarizer()
long_text = """
Climate change is one of the most pressing issues facing our planet today. It refers to long-term shifts in temperatures and weather patterns, mainly caused by human activities, especially the burning of fossil fuels. These activities release greenhouse gases into the atmosphere, trapping heat and causing the Earth's average temperature to rise. The consequences of climate change are far-reaching and include more frequent and severe weather events, rising sea levels, and disruptions to ecosystems. To address this global challenge, countries around the world are working to reduce their carbon emissions, transition to renewable energy sources, and implement policies to mitigate the effects of climate change. However, tackling this issue requires collective action from governments, businesses, and individuals alike.
"""

summary = summarizer.hybrid_summarize(long_text)
print("Original text length:", len(long_text))
print("Summary length:", len(summary))
print("Summary:", summary)
```

## 12.2 LLM 辅助机器翻译

### 12.2.1 翻译质量提升

LLM可以用于改善机器翻译的质量，特别是在处理上下文和语言细微差别方面。

```python
from transformers import MarianMTModel, MarianTokenizer
import torch

class EnhancedTranslator:
    def __init__(self, src_lang="en", tgt_lang="fr"):
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.llm = pipeline("text-generation", model="gpt2")

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

    def post_process(self, translation):
        prompt = f"Improve the following translation, making it more natural and fluent:\n{translation}\n\nImproved translation:"
        improved = self.llm(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        return improved.split("Improved translation:")[-1].strip()

    def enhanced_translate(self, text):
        basic_translation = self.translate(text)
        enhanced_translation = self.post_process(basic_translation)
        return enhanced_translation

# 使用示例
translator = EnhancedTranslator()
english_text = "The early bird catches the worm, but the second mouse gets the cheese."
french_translation = translator.enhanced_translate(english_text)
print(f"Original: {english_text}")
print(f"Enhanced Translation: {french_translation}")
```

### 12.2.2 多语言翻译系统

LLM可以用于构建更灵活的多语言翻译系统，处理多种语言之间的翻译。

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

class MultilingualTranslator:
    def __init__(self):
        self.model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
        self.tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")

    def translate(self, text, src_lang, tgt_lang):
        self.tokenizer.src_lang = src_lang
        encoded = self.tokenizer(text, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded,
            forced_bos_token_id=self.tokenizer.get_lang_id(tgt_lang)
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# 使用示例
translator = MultilingualTranslator()
text = "Hello, how are you?"
print(f"English: {text}")
print(f"French: {translator.translate(text, 'en', 'fr')}")
print(f"German: {translator.translate(text, 'en', 'de')}")
print(f"Spanish: {translator.translate(text, 'en', 'es')}")
```

### 12.2.3 专业领域翻译

LLM可以通过微调来适应特定领域的翻译需求，提高专业术语的翻译准确性。

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
import torch

class DomainSpecificTranslator:
    def __init__(self, base_model="Helsinki-NLP/opus-mt-en-fr"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)

    def fine_tune(self, train_data, eval_data, output_dir="./fine_tuned_translator"):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=eval_data
        )

        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def translate(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        translated = self.model.generate(**inputs)
        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

# 使用示例（假设我们有医学领域的平行语料库）
medical_translator = DomainSpecificTranslator()

# 准备训练数据（这里只是示例，实际应用中需要大量的领域特定平行语料）
train_data = [
    ("The patient presents with acute myocardial infarction.", "Le patient présente un infarctus aigu du myocarde."),
    ("Administer 325mg of aspirin stat.", "Administrer 325 mg d'aspirine immédiatement.")
]
eval_data = [
    ("Check for signs of cerebral ischemia.", "Vérifier les signes d'ischémie cérébrale.")
]

# 微调模型
medical_translator.fine_tune(train_data, eval_data)

# 使用微调后的模型进行翻译
medical_text = "The patient shows symptoms of severe pneumonia."
translated_text = medical_translator.translate(medical_text)
print(f"Original: {medical_text}")
print(f"Translated: {translated_text}")
```

## 12.3 LLM 在文本分类中的应用

### 12.3.1 零样本分类

LLM的强大语言理解能力使得零样本分类成为可能，无需针对特定任务进行训练。

```python
from transformers import pipeline

class ZeroShotClassifier:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify(self, text, candidate_labels):
        result = self.classifier(text, candidate_labels)
        return result['labels'][0], result['scores'][0]

# 使用示例
classifier = ZeroShotClassifier()
text = "The new iPhone has a stunning display and impressive camera quality."
labels = ["technology", "sports", "politics", "entertainment"]

top_label, confidence = classifier.classify(text, labels)
print(f"Text: {text}")
print(f"Top Label: {top_label}")
print(f"Confidence: {confidence:.2f}")
```

### 12.3.2 小样本学习

LLM可以通过少量示例快速适应新的分类任务，这在数据稀缺的情况下特别有用。

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class FewShotClassifier:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def create_prompt(self, examples, text):
        prompt = "Classify the following text into categories:\n\n"
        for example, label in examples:
            prompt += f"Text: {example}\nCategory: {label}\n\n"
        prompt += f"Text: {text}\nCategory:"
        return prompt

    def classify(self, text, examples, possible_labels):
        prompt = self.create_prompt(examples, text)
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=input_ids.shape[1] + 20, num_return_sequences=1)
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        predicted_label = generated_text.split("Category:")[-1].strip()
        
        if predicted_label not in possible_labels:
            return max(possible_labels, key=lambda label: predicted_label.lower().count(label.lower()))
        return predicted_label

# 使用示例
classifier = FewShotClassifier()

examples = [
    ("The stock market saw a significant drop today.", "Finance"),
    ("Scientists discover a new species of deep-sea fish.", "Science"),
    ("The latest superhero movie broke box office records.", "Entertainment")
]

text_to_classify = "A new study shows the benefits of regular exercise on mental health."
possible_labels = ["Finance", "Science", "Entertainment", "Health"]

prediction = classifier.classify(text_to_classify, examples, possible_labels)
print(f"Text: {text_to_classify}")
print(f"Predicted Category: {prediction}")
```

### 12.3.3 多标签分类优化

LLM可以用于处理复杂的多标签分类任务，捕捉文本的多个方面。

```python
from transformers import pipeline

class MultiLabelClassifier:
    def __init__(self):
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    def classify_multi_label(self, text, labels, threshold=0.5):
        results = self.classifier(text, labels, multi_label=True)
        return [(label, score) for label, score in zip(results['labels'], results['scores']) if score > threshold]

# 使用示例
classifier = MultiLabelClassifier()
text = "The new electric car combines cutting-edge technology with environmental sustainability, offering both performance and eco-friendliness."
labels = ["Technology", "Environment", "Automotive", "Innovation", "Sustainability"]

predictions = classifier.classify_multi_label(text, labels)
print(f"Text: {text}")
print("Predicted Labels:")
for label, score in predictions:
    print(f"- {label}: {score:.2f}")
```

通过将LLM与传统NLP技术相结合，我们可以显著提升各种文本处理任务的性能。这种融合不仅利用了LLM的强大语言理解能力，还保留了传统方法的优势，如效率和可解释性。在实际应用中，可能需要根据具体任务和资源限制来选择最合适的方法组合。
