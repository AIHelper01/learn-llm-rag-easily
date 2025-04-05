import streamlit
import os


# 数据库选型
from langchain.vectorstores.chroma import Chroma

# llm选型
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
# zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
# llm_deepseek = ChatOpenAI(
#     temperature=0.1,
#     model="glm-4",
#     openai_api_key=zhipuai_api_key,
#     openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
# )

from langchain_deepseek import ChatDeepSeek
deepseek_api_key = os.environ['DEEPSEEK_API_KEY']
llm_deepseek = ChatDeepSeek(model="deepseek-chat",api_key=deepseek_api_key)

# from langchain_ollama import OllamaLLM
# llm_deepseek = OllamaLLM(base_url='http://localhost:11434', model='deepseek-r1:1.5b',temperature=0)

# from langchain_ollama import OllamaLLM
# llm_deepseek = OllamaLLM(base_url='http://localhost:11434', model='qwen2.5:0.5b',temperature=0.1)
# from langchain_community.llms import Ollama
# llm_deepseek = Ollama(base_url='http://localhost:11434', model='deepseek-r1:1.5b', temperature=0.1)

# embeddinig选型
## bge-m3
from langchain_community.embeddings import OllamaEmbeddings
emb_bgem3 = OllamaEmbeddings(base_url='http://localhost:11434',model="bge-m3:latest")

## ZHIPUAI_API
# zhipuai_api_key = os.environ['ZHIPUAI_API_KEY']
# from langchain_community.embeddings import ZhipuAIEmbeddings
# emb_bgem3 = ZhipuAIEmbeddings(model="embedding-2",api_key=zhipuai_api_key)

# rerank选型
import cohere
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())    # read local .env file
cohere_api_key = os.environ['COHERE_API_KEY']
cohere_client = cohere.Client(api_key=cohere_api_key)

from langchain_core.output_parsers import StrOutputParser
def generate_response(input_text):
    # llm = ChatOpenAI(temperature=0.7, zhipuai_api_key=zhipuai_api_key)
    # llm = ZhipuAILLM(model="glm-4", temperature=0, api_key=zhipuai_api_key)
    llm = llm_deepseek
    output = llm.invoke(input_text)
    output_parser = StrOutputParser()
    output = output_parser.invoke(output)
    return output

def get_vectordb():
    # 定义 Embeddings
    embedding = emb_bgem3
    # 定义持久化目录
    persist_directory = '../chroma-vmax'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
        collection_name="vmax-s",
        embedding_function=emb_bgem3
    )
    return vectordb

# rerank选型
from langchain.retrievers.document_compressors import CohereRerank
from dotenv import load_dotenv, find_dotenv
import cohere
_ = load_dotenv(find_dotenv())    # read local .env file
cohere_api_key = os.environ['COHERE_API_KEY']
cohere_client = cohere.Client(api_key=cohere_api_key)
compressor = CohereRerank(
    client=cohere_client,
    top_n=5,
    model="rerank-multilingual-v3.0"  # 支持多语言的新版本
)


# 不带记忆的
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

def get_qa_chain_without_memory(question:str):
    vectordb = get_vectordb()
    # llm = llm_deepseek
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=template)
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
        llm=llm_deepseek,
        retriever=compression_retriever,  # 替换为压缩检索器
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": QA_CHAIN_PROMPT,
            # "llm_kwargs": {"max_length": 300}  # 新增输出长度限制
        }
    )
    result = qa_chain({"query": question})
    return result["result"]

#带有历史记录的问答链
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
def get_qa_chain_with_memory(question:str):
    vectordb = get_vectordb()
    # llm = llm_deepseek

    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    """

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],
                                     template=template)

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # 与 prompt 的输入变量保持一致。
        return_messages=True  # 将以消息列表的形式返回聊天记录，而不是单个字符串
    )
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
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm_deepseek,
        retriever=compression_retriever,
        memory=memory,
        # return_source_documents=True,
        output_key="answer",  # 明确指定存储到内存的键
        combine_docs_chain_kwargs={  # 替代chain_type_kwargs
            "prompt": QA_CHAIN_PROMPT
        },
        verbose=True,  # 独立传递verbose参数
    )
    result = qa_chain({"question": question})
    return result['answer']

# Streamlit 应用程序界面
def main():
    streamlit.title('🔍📜🔧DeepSeek VMAX-S知识助手')
    # zhipuai_api_key = st.sidebar.text_input('GLM API Key', type='password')

    # 添加一个选择按钮来选择不同的模型
    #selected_method = st.sidebar.selectbox("选择模式", ["qa_chain", "chat_qa_chain", "None"])
    selected_method = streamlit.radio(
        "你想选择哪种模式进行对话？",
        ["None", "get_qa_chain_without_memory", "get_qa_chain_with_memory"],
        captions = ["不使用检索问答的普通模式", "不带历史记录的检索问答模式", "带历史记录的检索问答模式"])

    # 用于跟踪对话历史
    if 'messages' not in streamlit.session_state:
        streamlit.session_state.messages = []

    messages = streamlit.container(height=300)
    if prompt := streamlit.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        streamlit.session_state.messages.append({"role": "user", "text": prompt})

        if selected_method == "None":
            # 调用 respond 函数获取回答
            answer = generate_response(prompt)
        elif selected_method == "get_qa_chain_without_memory":
            answer = get_qa_chain_without_memory(prompt)
        elif selected_method == "get_qa_chain_with_memory":
            answer = get_qa_chain_with_memory(prompt)

        # 检查回答是否为 None
        if answer is not None:
            # 将LLM的回答添加到对话历史中
            streamlit.session_state.messages.append({"role": "assistant", "text": answer})

        # 显示整个对话历史
        for message in streamlit.session_state.messages:
            if message["role"] == "user":
                messages.chat_message("user").write(message["text"])
            elif message["role"] == "assistant":
                messages.chat_message("assistant").write(message["text"])   


if __name__ == "__main__":
    main()
