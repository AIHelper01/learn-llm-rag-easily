import streamlit as st
# from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import sys
from zhipuai_embedding import ZhipuAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

def get_llm():
    return OllamaLLM(base_url='http://localhost:11434', model='deepseek-r1:14b', temperature=0.1, streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

def get_emd()
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

#不带历史记录的问答链
def get_qa_chain(question:str):
    vectordb = get_vectordb()
    my_llm = Ollama(base_url='http://localhost:11434', model='deepseek-r1:1.5b', temperature=0.1)
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

#带有历史记录的问答链
def get_chat_qa_chain(question:str):
    vectordb = get_vectordb()
    my_llm = get_llm()
    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
    
    template = """你是VMAX运维助手，基于以下对话历史和上下文知识，用中文回答用户的问题。
    历史对话记录：
    {chat_history}
    
    上下文知识：
    {context}
    
    当前问题：{question}
    
    回答要求：
    1. 如果问题需要专业领域知识，优先使用上下文内容
    2. 若答案不在知识库中，明确告知"根据已知信息无法回答"
    3. 结尾添加"是否需要进一步说明？"[2,7](@ref)
    """
    
    # 创建包含变量占位的PromptTemplate
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})  # 控制检索文档数量
    
    # 修改链配置，注入自定义模板
    qa = ConversationalRetrievalChain.from_llm(
        my_llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},  # 关键参数绑定模板
        get_chat_history=lambda h: h  # 保持历史记录原始格式[4](@ref)
        )
    result = qa_chain({"query": question})
    return result["result"]



# Streamlit 应用程序界面
def main():
    st.title('🦜🔗 动手学大模型应用开发')
    # zhipuai_api_key = st.sidebar.text_input('GLM API Key', type='password')

    # 添加一个选择按钮来选择不同的模型
    #selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "qa_chain", "chat_qa_chain"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 用于跟踪对话历史
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    messages = st.container(height=300)
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt)
        elif selected_method == "qa_chain":
            answer = get_qa_chain(prompt)
        elif selected_method == "chat_qa_chain":
            answer = get_chat_qa_chain(prompt)

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
