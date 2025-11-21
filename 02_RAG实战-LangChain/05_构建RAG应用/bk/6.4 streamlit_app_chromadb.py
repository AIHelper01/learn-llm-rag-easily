from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import cohere
from langchain_deepseek import ChatDeepSeek
import os
from dotenv import load_dotenv, find_dotenv


# 初始化函数 (保持你的原始代码不变)
def get_vectordb():
    emb_bgem3 = OllamaEmbeddings(base_url='http://localhost:11434', model="bge-m3:latest")
    persist_directory = '../chroma-vmax'
    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name="vmax-s",
        embedding_function=emb_bgem3
    )
    return vectordb


def get_llm():
    return OllamaLLM(base_url='http://localhost:11434', model='deepseek-r1:1.5b', temperature=0.1)


def generate_response(input_text):
    llm = get_llm()
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    return output_parser.invoke(output)


def get_qa_chain_without_memory(question: str):
    vectordb = get_vectordb()
    myllm = get_llm()

    cohere_client = cohere.Client(api_key="Tahx1eySFbKvu9sTyTXrRLf59la3ZUG9vy02stRZ")
    compressor = CohereRerank(
        client=cohere_client,
        top_n=5,
        model="rerank-multilingual-v3.0"
    )

    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": 15},
        search_type="mmr",
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=myllm,
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                input_variables=["context", "question"],
                template="""你是DeepSeek VMAX-S知识助手。使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
                案。总是在回答的最后说“谢谢你的提问！”。
                {context}
                问题: {question}
                """
            ),
        }
    )

    result = qa_chain({"query": question})
    return result["result"]


def get_qa_chain_with_memory(question: str):
    vectordb = get_vectordb()
    myllm = get_llm()

    cohere_client = cohere.Client(api_key="Tahx1eySFbKvu9sTyTXrRLf59la3ZUG9vy02stRZ")
    compressor = CohereRerank(
        client=cohere_client,
        top_n=5,
        model="rerank-multilingual-v3.0"
    )

    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": 15},
        search_type="mmr",
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # QA_CHAIN_PROMPT = PromptTemplate(
    #     input_variables=["chat_history", "question", "context"],
    #     template="""
    #     你是一个专业的问答助手。请根据对话历史和提供的上下文回答问题。

    #     历史对话：
    #     {chat_history}

    #     上下文：
    #     {context}

    #     问题：{question}

    #     回答：
    #     """
    # )

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template="""你是DeepSeek VMAX-S知识助手。使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。总是在回答的最后说“谢谢你的提问！”。
                {context}
                问题: {question}
        """
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=compression_retriever,
        memory=memory,
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": QA_CHAIN_PROMPT
        },
        verbose=True,
    )

    result = qa_chain({"question": question})  # Changed from "query" to "question"
    return result

# print(get_qa_chain_without_memory("介绍下VMAX的上网日志业务"))




# Initialize memory outside the function so it persists across questions
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


def get_vectordb():
    emb_bgem3 = OllamaEmbeddings(base_url='http://localhost:11434', model="bge-m3:latest")
    persist_directory = '../chroma-vmax'
    vectordb = Chroma(
        persist_directory=persist_directory,
        collection_name="vmax-s",
        embedding_function=emb_bgem3
    )
    return vectordb


def get_llm():
    return OllamaLLM(base_url='http://localhost:11434', model='deepseek-r1:1.5b', temperature=0.1)


def get_qa_chain_with_memory(question: str):
    vectordb = get_vectordb()
    myllm = get_llm()

    cohere_client = cohere.Client(api_key="Tahx1eySFbKvu9sTyTXrRLf59la3ZUG9vy02stRZ")
    compressor = CohereRerank(
        client=cohere_client,
        top_n=5,
        model="rerank-multilingual-v3.0"
    )

    base_retriever = vectordb.as_retriever(
        search_kwargs={"k": 15},
        search_type="mmr",
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template="""
        你是DeepSeek VMAX-S知识助手。使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。总是在回答的最后说“谢谢你的提问！

        历史对话：
        {chat_history}

        上下文：
        {context}

        问题：{question}

        回答：
        """
    )

    # QA_CHAIN_PROMPT = PromptTemplate(
    #     input_variables=["context","question"],
    #     template="""你是DeepSeek VMAX-S知识助手。使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答案。总是在回答的最后说“谢谢你的提问！”。
    #             {context}
    #             问题: {question}
    #     """
    # )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=compression_retriever,
        memory=memory,
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": QA_CHAIN_PROMPT
        },
        verbose=True,
    )

    result = qa_chain({"question": question})  # Changed from "query" to "question"
    return result


# questions = [
#     "VMAX上网日志业务包含哪些功能？",  # 需记忆前一轮的"主要内容"
#     "整理成markdown格式excel表格"  # 需合并多轮信息
# ]
#
# for question in questions:
#     result = get_qa_chain_with_memory(question)  # Pass string directly, not dict
#     print(f"问题：{question}")
#     print(f"回答：{result['answer']}")
#     print("对话历史：", memory.load_memory_variables({}))
#     print("\n" + "=" * 50 + "\n")

import streamlit as st

# 页面配置
st.set_page_config(page_title="DeepSeek VMAX 知识助手", page_icon="🤖")
st.title("DeepSeek VMAX 知识助手")
st.markdown("""
    ​**​三种模式​**​：
    - 🚀 直接生成：LLM 自由发挥（无检索）
    - 🔍 单次问答：基于知识库检索回答（无记忆）
    - 💬 连续对话：保留历史上下文的检索问答
""")

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "直接生成"

# 侧边栏配置
with st.sidebar:
    st.header("配置")
    mode = st.radio(
        "模式选择",
        ["直接生成", "单次问答", "连续对话"],
        index=["直接生成", "单次问答", "连续对话"].index(st.session_state.get("mode", "直接生成"))
    )

    # 动态显示参数
    if mode != "直接生成":
        st.subheader("检索参数")
        search_k = st.slider("检索文档数量 (k)", 1, 20, 15)
        rerank_top_n = st.slider("重排序保留数 (top_n)", 1, 10, 5)

    st.subheader("模型参数")
    temperature = st.slider("温度 (temperature)", 0.0, 1.0, 0.1, 0.05)

    if st.button("清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# 模式切换时清空历史（避免上下文混淆）
if st.session_state.get("mode") != mode:
    st.session_state.messages = []
    st.session_state.mode = mode

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 用户输入处理
if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 生成回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                if mode == "直接生成":
                    response = generate_response(prompt)
                elif mode == "单次问答":
                    response = get_qa_chain_without_memory(prompt)
                else:
                    result = get_qa_chain_with_memory(prompt)
                    response = result.get("answer", "未能生成回答")

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"⚠️ 错误：{str(e)}")
                st.session_state.messages.append({"role": "assistant", "content": f"出错: {str(e)}"})