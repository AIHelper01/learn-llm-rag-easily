#!/usr/bin/env python
# coding: utf-8

# # 使用OpenAI API

# 
# GPT有封装好的接口，我们简单封装即可。目前GPT embedding mode有三种，性能如下所示：
# |模型 | 每美元页数 | [MTEB](https://github.com/embeddings-benchmark/mteb)得分 | [MIRACL](https://github.com/project-miracl/miracl)得分|
# | --- | --- | --- | --- |
# |text-embedding-3-large|9,615|54.9|64.6|
# |text-embedding-3-small|62,500|62.3|44.0|
# |text-embedding-ada-002|12,500|61.0|31.4|
# * MTEB得分为embedding model分类、聚类、配对等八个任务的平均得分。
# * MIRACL得分为embedding model在检索任务上的平均得分。  
# 
# 从以上三个embedding model我们可以看出`text-embedding-3-large`有最好的性能和最贵的价格，当我们搭建的应用需要更好的表现且成本充足的情况下可以使用；`text-embedding-3-small`有着较好的性能跟价格，当我们预算有限时可以选择该模型；而`text-embedding-ada-002`是OpenAI上一代的模型，无论在性能还是价格都不如及前两者，因此不推荐使用。

# In[1]:


import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 如果你需要通过代理端口访问，你需要如下配置
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

def openai_embedding(text: str, model: str=None):
    # 获取环境变量 OPENAI_API_KEY
    api_key=os.environ['OPENAI_API_KEY']
    client = OpenAI(api_key=api_key)

    # embedding model：'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
    if model == None:
        model="text-embedding-3-small"

    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response

response = openai_embedding(text='要生成 embedding 的输入文本，字符串形式。')


# API返回的数据为`json`格式，除`object`向量类型外还有存放数据的`data`、embedding model 型号`model`以及本次 token 使用情况`usage`等数据，具体如下所示：
# ```json
# {
#   "object": "list",
#   "data": [
#     {
#       "object": "embedding",
#       "index": 0,
#       "embedding": [
#         -0.006929283495992422,
#         ... (省略)
#         -4.547132266452536e-05,
#       ],
#     }
#   ],
#   "model": "text-embedding-3-small",
#   "usage": {
#     "prompt_tokens": 5,
#     "total_tokens": 5
#   }
# }
# ```
# 我们可以调用response的object来获取embedding的类型。

# In[2]:


print(f'返回的embedding类型为：{response.object}')


# embedding存放在data中，我们可以查看embedding的长度及生成的embedding。

# In[3]:


print(f'embedding长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为：{response.data[0].embedding[:10]}')


# 我们也可以查看此次embedding的模型及token使用情况。

# In[4]:


print(f'本次embedding model为：{response.model}')
print(f'本次token使用情况为：{response.usage}')


# In[5]:


import requests
import json

def wenxin_embedding(text: str):
    # 获取环境变量 wenxin_api_key、wenxin_secret_key
    api_key = os.environ['QIANFAN_AK']
    secret_key = os.environ['QIANFAN_SK']

    # 使用API Key、Secret Key向https://aip.baidubce.com/oauth/2.0/token 获取Access token
    url = "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id={0}&client_secret={1}".format(api_key, secret_key)
    payload = json.dumps("")
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)

    # 通过获取的Access token 来embedding text
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/embedding-v1?access_token=" + str(response.json().get("access_token"))
    input = []
    input.append(text)
    payload = json.dumps({
        "input": input
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return json.loads(response.text)
# text应为List(string)
text = "要生成 embedding 的输入文本，字符串形式。"
response = wenxin_embedding(text=text)


# Embedding-V1每次embedding除了有单独的id外，还有时间戳记录embedding的时间。

# In[6]:


print('本次embedding id为：{}'.format(response['id']))
print('本次embedding产生时间戳为：{}'.format(response['created']))


# 同样的我们也可以从response中获取embedding的类型和embedding。

# In[7]:


print('返回的embedding类型为:{}'.format(response['object']))
print('embedding长度为：{}'.format(len(response['data'][0]['embedding'])))
print('embedding（前10）为：{}'.format(response['data'][0]['embedding'][:10]))


# 尚未开放

# # 使用智谱API
# 智谱有封装好的SDK，我们调用即可。

# In[2]:


import os

from dotenv import load_dotenv, find_dotenv

# 读取本地/项目的环境变量。
# find_dotenv() 寻找并定位 .env 文件的路径
# load_dotenv() 读取该 .env 文件，并将其中的环境变量加载到当前的运行环境中  
_ = load_dotenv(find_dotenv())


# In[3]:


import os
from zhipuai import ZhipuAI
def zhipu_embedding(text: str):
    api_key = os.environ['ZHIPUAI_API_KEY']
    client = ZhipuAI(api_key=api_key)
    response = client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return response

text = '这是一个测试句子。'
response = zhipu_embedding(text=text)


# response为`zhipuai.types.embeddings.EmbeddingsResponded`类型，我们可以调用`object`、`data`、`model`、`usage`来查看response的embedding类型、embedding、embedding model及使用情况。

# In[4]:


print(f'response类型为：{type(response)}')
print(f'embedding类型为：{response.object}')
print(f'生成embedding的model为：{response.model}')
print(f'生成的embedding长度为：{len(response.data[0].embedding)}')
print(f'embedding（前10）为: {response.data[0].embedding[:10]}')


# In[7]:


from langchain_community.embeddings import ZhipuAIEmbeddings

zhipu_embed = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key="5713143e8fdc4b4a8b284cf97092e70f.qEK71mGIlavzO1Io",
)
r1 = zhipu_embed.embed_documents(
    [
        "Alpha is the first letter of Greek alphabet",
        "Beta is the second letter of Greek alphabet",
    ]
)
r2 = zhipu_embed.embed_query(
    "What is the second letter of Greek alphabet"
)

# 打印结果
print("Document embeddings:")
for i, embedding in enumerate(r1):
    print(f"Document {i+1} embedding: {embedding[:10]}")

print("\nQuery embedding:")
print(f"Query embedding: {r2[:10]}")


# # 使用HuggingFaceEmbeddings

# 使用 langchain_community 库中的 HuggingFaceEmbeddings 类来加载一个预训练的模型 "sentence-transformers/all-mpnet-base-v2"。
# 这个模型是基于 MPNet 架构的，能够将文本转换为密集向量（即嵌入），这些向量可以用于各种自然语言处理任务，如语义相似度计算、文本分类等。

# In[ ]:


from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


# model_name: 指定了要使用的预训练模型的名称。这里使用的是 sentence-transformers/all-mpnet-base-v2，这是一个非常强大的句子编码器。  
# model_kwargs: 这是一个字典，可以用来传递额外的关键字参数给模型。在这里，你指定了设备为 'cpu'，这意味着模型将在 CPU 上运行而不是 GPU。  
# encode_kwargs: 这个参数允许你在编码过程中指定额外的行为。例如，normalize_embeddings=False 表示生成的嵌入不会被归一化。  

# In[3]:


# 输出与输入句子相对应的嵌入向量
text = "这是一个测试句子。"
embedding = hf.embed_query(text)
print(embedding[:10])


# In[5]:


# 如果有多个文档需要嵌入，可以使用 embed_documents 方法
texts = ["这是第一个句子。", "这是第二个句子。"]
embeddings = hf.embed_documents(texts)
for i, emb in enumerate(embeddings):
    print(f"句子 {i+1} 的嵌入: {emb[:10]}")


# # 使用ollama Embeddings

# In[6]:


from langchain_community.embeddings import OllamaEmbeddings
ollama_emb = OllamaEmbeddings(
    base_url='http://localhost:11434',
    model="nomic-embed-text:latest"
)
r1 = ollama_emb.embed_documents(
    [
        "Alpha is the first letter of Greek alphabet",
        "Beta is the second letter of Greek alphabet",
    ]
)
r2 = ollama_emb.embed_query(
    "What is the second letter of Greek alphabet"
)


# In[7]:


# 打印结果
print("Document embeddings:")
for i, embedding in enumerate(r1):
    print(f"Document {i+1} embedding: {embedding[:10]}")

print("\nQuery embedding:")
print(f"Query embedding: {r2[:10]}")


# In[ ]:





# # xinference embedding

# In[ ]:




