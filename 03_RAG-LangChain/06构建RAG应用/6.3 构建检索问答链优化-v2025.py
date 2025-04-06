from langchain_chroma import Chroma
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pymilvus import Collection, connections
import re

# Initialize memory outside the function so it persists across questions
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 初始化 Milvus 向量数据库
def get_vectordb():
    emb_bgem3 = OllamaEmbeddings(base_url='http://localhost:11434', model="bge-m3:latest")

    # Milvus 连接参数
    vectordb = Milvus(
        embedding_function=emb_bgem3,
        collection_name="Vmaxs",  # Milvus 集合名称
        connection_args={
            "host": "192.168.0.188",  # Milvus 服务器地址
            "port": "19530",  # Milvus 默认端口
        },
    )
    return vectordb

def get_llm():
    return OllamaLLM(base_url='http://localhost:11434', model='deepseek-r1:1.5b', temperature=0.1, streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()])

def get_text_list_from_milvus(
        collection_name: str,
        host: str = "192.168.0.188",
        port: str = "19530",
        expr: str = "",
        limit: int = 1000,
        output_fields: list = ["text"],
) -> list:
    """
    从 Milvus 集合中读取指定字段（默认是 text）并返回列表
    """
    # 1. 连接 Milvus
    connections.connect(alias="default", host=host, port=port)

    # 2. 加载集合
    collection = Collection(name=collection_name)
    collection.load()

    # 3. 查询数据
    results = collection.query(
            expr=expr,
            output_fields=output_fields,
            limit=limit
        )

    # 4. 提取目标字段为列表
    if not output_fields:
        raise ValueError("output_fields 不能为空")

    field_name = output_fields[0]  # 默认取第一个字段
    data_list = [item[field_name] for item in results]
    return data_list

def determine_query_type(question: str) -> str:
    """
    根据问题内容判断查询类型，返回权重调整策略

    返回:
        "keyword" - 更适合关键词检索的问题
        "semantic" - 更适合语义检索的问题
        "balanced" - 平衡型问题
    """
    # 关键词型问题特征
    keyword_patterns = [
        r"什么是.*\?",  # 定义类问题
        r".*包括哪些.*",  # 列举类问题
        r".*有哪些.*",  # 列举类问题
        r".*多少种.*",  # 数量类问题
        r".*步骤.*",  # 流程类问题
        r".*如何.*",  # 方法类问题
        r".*怎样.*",  # 方法类问题
        r".*整理.*表格",  # 结构化输出要求
        r".*列出.*",  # 列举要求
        r".*对比.*",  # 比较类问题
    ]

    # 语义型问题特征
    semantic_patterns = [
        r".*解决.*问题",  # 解决方案类
        r".*原因.*",  # 原因分析类
        r".*为什么.*",  # 原因分析类
        r".*建议.*",  # 建议类
        r".*优缺点.*",  # 分析类
        r".*影响.*",  # 影响分析类
        r".*解释.*",  # 解释说明类
        r".*理解.*",  # 理解类
        r".*意味着什么",  # 含义类
    ]

    # 检查是否是关键词型问题
    for pattern in keyword_patterns:
        if re.search(pattern, question):
            return "keyword"

    # 检查是否是语义型问题
    for pattern in semantic_patterns:
        if re.search(pattern, question):
            return "semantic"

    # 默认平衡型
    return "balanced"


def get_dynamic_weights(query_type: str) -> tuple:
    """
    根据查询类型返回动态权重

    返回:
        tuple: (bm25_weight, vector_weight)
    """
    if query_type == "keyword":
        return (0.7, 0.3)  # 更侧重关键词检索
    elif query_type == "semantic":
        return (0.3, 0.7)  # 更侧语义检索
    else:
        return (0.5, 0.5)  # 平衡权重

def get_qa_chain_with_memory(question: str):
    vectordb = get_vectordb()

    # 1. 确定查询类型和动态权重
    query_type = determine_query_type(question)
    bm25_weight, vector_weight = get_dynamic_weights(query_type)
    print(f"问题类型: {query_type}, 权重设置: BM25={bm25_weight}, Vector={vector_weight}")

    # 2. 初始化 BM25 检索器
    documents = get_text_list_from_milvus(collection_name="Vmaxs")
    bm25_retriever = BM25Retriever.from_texts(documents)
    bm25_retriever.k = 10

    # 3. 初始化向量检索器
    vector_retriever = vectordb.as_retriever(
        search_kwargs={"k": 10},
        search_type="mmr",
    )

    # 4. 使用动态权重的混合检索
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],  # 使用动态权重
    )
    # 5. 混合检索（EnsembleRetriever）作为最终检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],  # 调整BM25和向量检索的权重
    )

   # 6. 定义提示模板（加入权重信息）
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=f"""
    你是一位专业的VMAX-S技术专家助手，负责回答关于VMAX-S产品的技术问题。请根据以下规则回答问题：

    1. 严格基于提供的上下文信息回答，不要编造或假设
    2. 如果上下文不包含答案，明确表示"根据现有资料，我无法回答这个问题"
    3. 回答要专业、准确、简洁
    4. 对于操作类问题，提供分步骤说明
    5. 对于需要比较或列举的问题，使用表格或列表形式
    6. 保持友好专业的语气

    当前问题类型: {query_type} (检索权重: BM25={bm25_weight}, Vector={vector_weight})

    当前对话历史：
    {{chat_history}}

    相关技术资料：
    {{context}}

    用户问题：{{question}}

    请按照以下格式回答：
    [专业回答]
    (你的回答内容)

    [补充说明]
    (如有需要，可添加额外说明或建议)

    感谢您咨询VMAX-S相关问题！
    """
    )


    # 7. 构建对话式检索链
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=ensemble_retriever,  # 直接使用混合检索器
        memory=memory,
        output_key="answer",
        combine_docs_chain_kwargs={
            "prompt": QA_CHAIN_PROMPT
        },
        verbose=False
    )

    result = qa_chain({"question": question})
    return result

# 测试问题
questions = [
    "什么是VMAX的上网日志业务？",
    "上网日志业务包含哪些功能？"
]

for question in questions:
    print(f"\n问题: {question}")
    result = get_qa_chain_with_memory(question)
    print(f"\n回答: {result['answer']}")
    print("=" * 50)