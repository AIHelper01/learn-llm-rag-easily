#!/usr/bin/env python
# coding: utf-8



import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv


# 读取本地/项目的环境变量。
# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中  
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())



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



# In[ ]:





# # xinference embedding

# In[ ]:




