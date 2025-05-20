import streamlit as st
# from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Milvus
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings


def get_llm():
    return Ollama(base_url='http://localhost:11434', model='deepseek-r1:14b', temperature=0.1)

def get_emd():
    return OllamaEmbeddings(base_url='http://localhost:11434', model="bge-m3:latest")

# 初始化 Milvus 向量数据库
def get_vectordb():
    my_emb = get_emd()
    # Milvus 连接参数
    vectordb = Milvus(
        embedding_function=my_emb,
        collection_name="Vmaxs",  # Milvus 集合名称
        connection_args={
            "host": "192.168.0.188",  # Milvus 服务器地址
            "port": "19530",  # Milvus 默认端口
        },
    )
    return vectordb

# 不带知识库的回答
def generate_response(input_text):
    my_llm = get_llm()
    output = my_llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    return output

# 基于知识库的问答链
def generate_response_with_rag(question:str):
    vectordb = get_vectordb()
    my_llm = get_llm()
    template = """你是VMAX运维助手，使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                 template=template)
    qa_chain = RetrievalQA.from_chain_type(my_llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]




def generate_response_with_rag_memory(question: str):
    # 初始化向量数据库和LLM
    vectordb = get_vectordb()
    my_llm = get_llm()

    memory = ConversationBufferMemory( memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
    return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    
    # 修改后的Prompt模板（添加chat_history变量）
    template = """你是VMAX运维助手，请参考以下对话历史和上下文来回答问题：
    {chat_history}
    
    相关上下文：
    {context}
    
    问题：{question}
    回答结束时说“谢谢你的提问！”
    """
    
    QA_PROMPT = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template
    )
    
    # 创建对话链
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=my_llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        chain_type="stuff"
    )
    
    result = qa_chain({"question": question})
    return result["answer"]


# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 VMAX-S运维助手Demo')
    # zhipuai_api_key = st.sidebar.text_input('GLM API Key', type='password')

    # 添加一个选择按钮来选择不同的模型
    #selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["No-RAG", "generate_response_with_rag", "generate_response_with_rag_memory"],
        captions = ["不使用基于知识库的检索问答模式", "基于知识库的检索问答模式", "基于知识库的检索问答模式（带记忆）"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "No-RAG":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt)
        elif selected_method == "generate_response_with_rag":
            answer = generate_response_with_rag(prompt)
        elif selected_method == "generate_response_with_rag_memory":
            answer = generate_response_with_rag(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            st.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()
