
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

