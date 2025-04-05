# import sys
# sys.path.append("../C3 搭建知识库") # 将父目录放入系统路径中

# 使用智谱 Embedding API，注意，需要将上一章实现的封装代码下载到本地
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv, find_dotenv
import os
# 从环境变量中加载你的 API_KEY
_ = load_dotenv(find_dotenv())    # read local .env file
zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']

# 定义持久化目录
persist_directory = '../chroma-vmax-6'

# 创建嵌入模型
from langchain_community.embeddings import ZhipuAIEmbeddings

zhipu_embed = ZhipuAIEmbeddings(
    model="embedding-2",
    api_key=zhipuai_api_key
)

try:
    # 加载持久化的 Chroma 向量数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        collection_name="vmax-s",
        embedding_function=zhipu_embed
    )
    print("向量数据库已成功加载。")
except Exception as e:
    print(f"加载向量数据库时发生错误: {e}")


from langchain_openai import ChatOpenAI
zhipuai_llm = ChatOpenAI(
    temperature=0.95,
    model="glm-4",
    openai_api_key=zhipuai_api_key,
    openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)

from langchain.prompts import PromptTemplate

template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
{context}
问题: {question}
"""

QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
)


from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
# rerank检索
# Cohere Rerank配置
import cohere
cohere_client = cohere.Client(api_key="Tahx1eySFbKvu9sTyTXrRLf59la3ZUG9vy02stRZ")

compressor = CohereRerank(
    client=cohere_client,
    top_n=5,
    model="rerank-multilingual-v3.0"  # 支持多语言的新版本
)

base_retriever = vectordb.as_retriever(
    search_kwargs={"k": 15},  # 扩大召回池
    search_type="mmr",  # 最大边际相关性算法（网页5）
    # metadata_filter={"source": "权威文档.pdf"}  # 元数据过滤
)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)


qa_chain = RetrievalQA.from_chain_type(
    zhipuai_llm,
    retriever=compression_retriever,  # 替换为压缩检索器
    return_source_documents=True,
    chain_type_kwargs={
        "prompt": QA_CHAIN_PROMPT,
        # "llm_kwargs": {"max_length": 300}  # 新增输出长度限制
    }
)

from langchain.chains import ConversationalRetrievalChain

from langchain.prompts import PromptTemplate

QA_CHAIN_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template="""
    你是一个专业的问答助手。请根据对话历史和提供的上下文回答问题。

    历史对话：
    {chat_history}

    上下文：
    {context}

    问题：{question}

    回答：
    """
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=zhipuai_llm,
    retriever=compression_retriever,
    memory=memory,
    # return_source_documents=True,
    output_key="answer",  # 明确指定存储到内存的键
    combine_docs_chain_kwargs={  # 替代chain_type_kwargs
        "prompt": QA_CHAIN_PROMPT
    },
    verbose=True,  # 独立传递verbose参数
)

questions = [
    "什么是VMAX的安全加固？",
    "安全加固的操作步骤？",  # 需记忆前一轮的"主要内容"
    "整理成中文表格"  # 需合并多轮信息
]

for question in questions:
    result = qa_chain({"question": question})
    print(f"问题：{question}")
    print(f"回答：{result['answer']}")
    # print(f"引用的来源：{result['source_documents'][0].metadata}")  # 显示来源文档
    print("对话历史：", memory.load_memory_variables({}))