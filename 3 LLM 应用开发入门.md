
# 3 LLM 应用开发入门

## 3.1 第一个 LLM 应用：Hello World

### 3.1.1 连接 API

在开始开发我们的第一个LLM应用之前，我们需要确保能够成功连接到LLM API。以OpenAI API为例，我们将创建一个简单的连接测试函数。

```python
import openai
import os

def test_api_connection():
    # 从环境变量获取API密钥
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        # 尝试列出可用的模型
        models = openai.Model.list()
        print("成功连接到OpenAI API!")
        print(f"可用模型数量: {len(models.data)}")
        return True
    except Exception as e:
        print(f"连接OpenAI API时出错: {e}")
        return False

# 运行测试
if __name__ == "__main__":
    test_api_connection()
```

确保在运行此脚本之前，你已经设置了`OPENAI_API_KEY`环境变量。

### 3.1.2 发送简单请求

现在我们已经确认可以连接到API，让我们创建一个简单的"Hello World"应用。这个应用将向LLM发送一个提示，要求它生成一个问候语。

```python
import openai
import os

def generate_greeting(name):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=f"Generate a friendly greeting for {name}:",
            max_tokens=30
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Error generating greeting: {e}"

# 使用示例
if __name__ == "__main__":
    user_name = input("请输入你的名字: ")
    greeting = generate_greeting(user_name)
    print(greeting)
```

### 3.1.3 解析和展示结果

LLM API返回的结果通常是一个包含多个字段的JSON对象。让我们创建一个函数来解析这个结果，并以更友好的方式展示它。

```python
import openai
import os
import json

def generate_and_parse_response(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50
        )
        
        # 解析响应
        parsed_response = {
            "text": response.choices[0].text.strip(),
            "model": response.model,
            "usage": response.usage
        }
        
        return parsed_response
    except Exception as e:
        return {"error": str(e)}

def display_response(response):
    if "error" in response:
        print(f"错误: {response['error']}")
    else:
        print("生成的文本:")
        print(response["text"])
        print("\n模型信息:")
        print(f"使用的模型: {response['model']}")
        print(f"Token使用情况: {json.dumps(response['usage'], indent=2)}")

# 使用示例
if __name__ == "__main__":
    prompt = input("请输入一个提示: ")
    result = generate_and_parse_response(prompt)
    display_response(result)
```

这个示例展示了如何发送请求、解析响应，并以用户友好的方式展示结果。它还包括了错误处理，确保在API调用失败时能够优雅地处理错误。

通过这个"Hello World"应用，我们已经完成了LLM应用开发的基本流程：连接API、发送请求、解析响应和展示结果。这为我们后续开发更复杂的LLM应用奠定了基础。

## 3.2 基本操作和概念

### 3.2.1 Prompt 工程入门

Prompt工程是LLM应用开发中的关键技能。它涉及设计和优化输入提示，以获得最佳的模型输出。以下是一些基本的Prompt工程技巧：

1. 明确指令：提供清晰、具体的指示。
2. 上下文提供：给出相关背景信息。
3. 示例展示：通过few-shot learning提供示例。
4. 格式指定：明确指定所需的输出格式。

示例：改进问候语生成

```python
def improved_greeting_generator(name, tone, occasion):
    prompt = f"""
    Generate a greeting with the following specifications:
    - Name: {name}
    - Tone: {tone} (e.g., formal, friendly, humorous)
    - Occasion: {occasion} (e.g., birthday, job interview, casual meeting)
    
    Format: [Greeting], [Name]! [Occasion-specific message]
    
    Example:
    Name: Alice
    Tone: Friendly
    Occasion: Birthday
    Output: Hey, Alice! Hope your birthday is filled with joy and laughter!
    
    Now generate the greeting:
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50
    )
    
    return response.choices[0].text.strip()

# 使用示例
name = "John"
tone = "formal"
occasion = "job interview"
greeting = improved_greeting_generator(name, tone, occasion)
print(greeting)
```

### 3.2.2 温度和采样策略

温度是控制LLM输出随机性的参数。较低的温度会产生更确定和一致的输出，而较高的温度会增加创造性和多样性。

采样策略决定了如何从模型的概率分布中选择下一个标记。常见的策略包括：
- Greedy sampling：始终选择概率最高的标记
- Top-k sampling：从概率最高的k个标记中随机选择
- Top-p (nucleus) sampling：从累积概率超过p的最小标记集中选择

示例：探索不同温度的效果

```python
def generate_with_temperature(prompt, temperature):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        temperature=temperature
    )
    return response.choices[0].text.strip()

# 使用示例
prompt = "Write a short story about a robot learning to love:"
temperatures = [0.2, 0.5, 0.8]

for temp in temperatures:
    print(f"\nTemperature: {temp}")
    story = generate_with_temperature(prompt, temp)
    print(story)
```

### 3.2.3 Token 限制和管理

LLM API通常有token数量限制，包括输入和输出的总token数。有效管理token使用对于控制成本和确保响应质量至关重要。

Token计数示例：

```python
from transformers import GPT2Tokenizer

def count_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return len(tokenizer.encode(text))

# 使用示例
text = "Hello, world! This is a sample text for token counting."
token_count = count_tokens(text)
print(f"Token count: {token_count}")
```

Token管理策略：
1. 截断长输入
2. 分块处理长文本
3. 使用摘要技术压缩输入

示例：处理长文本

```python
def process_long_text(text, max_tokens=2000):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in text.split('.'):
        sentence_tokens = tokenizer.encode(sentence)
        if current_length + len(sentence_tokens) > max_tokens:
            chunks.append('.'.join(current_chunk))
            current_chunk = []
            current_length = 0
        
        current_chunk.append(sentence)
        current_length += len(sentence_tokens)
    
    if current_chunk:
        chunks.append('.'.join(current_chunk))
    
    return chunks

# 使用示例
long_text = "This is a very long text... (假设这是一个很长的文本)"
chunks = process_long_text(long_text)
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(chunk)
    print(f"Token count: {count_tokens(chunk)}\n")
```

## 3.3 简单应用案例

### 3.3.1 智能问答机器人

创建一个简单的问答机器人，能够回答用户的一般性问题。

```python
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def qa_bot(question):
    prompt = f"""
    Human: {question}
    AI: Let me think about that and provide a helpful answer.
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    
    return response.choices[0].text.strip()

# 交互式问答循环
while True:
    user_question = input("Ask a question (or type 'exit' to quit): ")
    if user_question.lower() == 'exit':
        break
    
    answer = qa_bot(user_question)
    print("AI:", answer)
    print()
```

### 3.3.2 文本分类器

使用LLM创建一个简单的文本分类器，可以对输入的文本进行情感分析或主题分类。

```python
def classify_text(text, categories):
    prompt = f"""
    Classify the following text into one of these categories: {', '.join(categories)}
    
    Text: "{text}"
    
    Classification:
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=10,
        temperature=0.3
    )
    
    return response.choices[0].text.strip()

# 使用示例
categories = ["Technology", "Sports", "Politics", "Entertainment"]
sample_texts = [
    "The new smartphone features a revolutionary AI chip.",
    "The team won the championship after a thrilling overtime victory.",
    "The senate passed the controversial bill with a narrow margin."
]

for text in sample_texts:
    classification = classify_text(text, categories)
    print(f"Text: {text}")
    print(f"Classification: {classification}\n")
```

### 3.3.3 简单的翻译工具

创建一个基于LLM的简单翻译工具，支持多种语言之间的翻译。

```python
def translate_text(text, source_lang, target_lang):
    prompt = f"""
    Translate the following {source_lang} text to {target_lang}:
    
    {text}
    
    Translation:
    """
    
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=100,
        temperature=0.3
    )
    
    return response.choices[0].text.strip()

# 使用示例
text_to_translate = "Hello, how are you? It's a beautiful day today."
source = "English"
target = "French"

translation = translate_text(text_to_translate, source, target)
print(f"Original ({source}): {text_to_translate}")
print(f"Translation ({target}): {translation}")
```

这些简单的应用案例展示了LLM在各种任务中的versatility。通过调整prompt、温度和其他参数，我们可以优化这些应用以获得更好的性能。在接下来的章节中，我们将探讨更高级的LLM应用开发技术和最佳实践。
