from langchain_chroma import Chroma
from langchain_community.vectorstores import Milvus
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import cohere
from pymilvus import connections, Collection
import re
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd
import time

# 初始化内存
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 初始化句子转换模型用于评估
sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


# 初始化 Milvus 向量数据库
def get_vectordb():
    emb_bgem3 = OllamaEmbeddings(base_url='http://localhost:11434', model="bge-m3:latest")

    vectordb = Milvus(
        embedding_function=emb_bgem3,
        collection_name="Vmaxs",
        connection_args={
            "host": "192.168.0.188",
            "port": "19530",
        },
    )
    return vectordb


def get_llm():
    return OllamaLLM(
        base_url='http://localhost:11434',
        model='deepseek-r1:1.5b',
        temperature=0.1,
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()]
    )


def get_text_list_from_milvus(
        collection_name: str,
        host: str = "192.168.0.188",
        port: str = "19530",
        expr: str = "",
        limit: int = 1000,
        output_fields: list = ["text"],
) -> list:
    connections.connect(alias="default", host=host, port=port)
    collection = Collection(name=collection_name)
    collection.load()
    results = collection.query(
        expr=expr,
        output_fields=output_fields,
        limit=limit
    )
    field_name = output_fields[0]
    data_list = [item[field_name] for item in results]
    return data_list


def determine_query_type(question: str) -> str:
    keyword_patterns = [
        r"什么是.*\?", r".*包括哪些.*", r".*有哪些.*", r".*多少种.*",
        r".*步骤.*", r".*如何.*", r".*怎样.*", r".*整理.*表格",
        r".*列出.*", r".*对比.*"
    ]

    semantic_patterns = [
        r".*解决.*问题", r".*原因.*", r".*为什么.*", r".*建议.*",
        r".*优缺点.*", r".*影响.*", r".*解释.*", r".*理解.*",
        r".*意味着什么"
    ]

    for pattern in keyword_patterns:
        if re.search(pattern, question):
            return "keyword"

    for pattern in semantic_patterns:
        if re.search(pattern, question):
            return "semantic"

    return "balanced"


def get_dynamic_weights(query_type: str) -> tuple:
    if query_type == "keyword":
        return (0.7, 0.3)
    elif query_type == "semantic":
        return (0.3, 0.7)
    else:
        return (0.5, 0.5)


def get_qa_chain_with_memory(question: str, evaluate_mode: bool = False):
    vectordb = get_vectordb()

    # 1. 确定查询类型和动态权重
    query_type = determine_query_type(question)
    bm25_weight, vector_weight = get_dynamic_weights(query_type)

    if not evaluate_mode:
        print(f"问题类型: {query_type}, 权重设置: BM25={bm25_weight}, Vector={vector_weight}")

    # 2. 初始化检索器
    documents = get_text_list_from_milvus(collection_name="Vmaxs")
    bm25_retriever = BM25Retriever.from_texts(documents)
    bm25_retriever.k = 10

    vector_retriever = vectordb.as_retriever(
        search_kwargs={"k": 10},
        search_type="mmr",
    )

    # 3. 混合检索
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight],
    )

    # 4. Cohere Rerank
    cohere_client = cohere.Client(api_key="Tahx1eySFbKvu9sTyTXrRLf59la3ZUG9vy02stRZ")
    compressor = CohereRerank(
        client=cohere_client,
        top_n=5,
        model="rerank-multilingual-v3.0"
    )

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    # 5. 提示模板
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template="""
        你是一位专业的VMAX-S技术专家助手，负责回答关于VMAX-S产品的技术问题。请根据以下规则回答问题：

        1. 严格基于提供的上下文信息回答，不要编造或假设
        2. 如果上下文不包含答案，明确表示"根据现有资料，我无法回答这个问题"
        3. 回答要专业、准确、简洁
        4. 对于操作类问题，提供分步骤说明
        5. 对于需要比较或列举的问题，使用表格或列表形式
        6. 保持友好专业的语气

        当前问题类型: {query_type} (检索权重: BM25={bm25_weight}, Vector={vector_weight})

        当前对话历史：
        {chat_history}

        相关技术资料：
        {context}

        用户问题：{question}

        请按照以下格式回答：
        [专业回答]
        (你的回答内容)

        [补充说明]
        (如有需要，可添加额外说明或建议)

        感谢您咨询VMAX-S相关问题！
        """.format(query_type=query_type, bm25_weight=bm25_weight, vector_weight=vector_weight)
    )

    # 6. 构建对话链
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=get_llm(),
        retriever=compression_retriever,
        memory=None if evaluate_mode else memory,
        output_key="answer",
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT},
        verbose=False
    )

    start_time = time.time()
    result = qa_chain({"question": question})
    response_time = time.time() - start_time

    if evaluate_mode:
        return {
            "answer": result["answer"],
            "query_type": query_type,
            "weights": (bm25_weight, vector_weight),
            "response_time": response_time,
            "retriever": compression_retriever
        }
    return result


# ================== 评估模块 ==================

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """使用句子嵌入计算语义相似度"""
    embeddings = sentence_model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]


def evaluate_retrieval(
        questions: List[str],
        reference_docs: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    评估检索效果
    """
    results = []

    for question in questions:
        # 获取每种检索方法的结果
        vectordb = get_vectordb()
        documents = get_text_list_from_milvus(collection_name="Vmaxs")

        # BM25检索
        bm25_retriever = BM25Retriever.from_texts(documents)
        bm25_retriever.k = 10
        bm25_docs = bm25_retriever.get_relevant_documents(question)

        # 向量检索
        vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
        vector_docs = vector_retriever.get_relevant_documents(question)

        # 混合检索
        query_type = determine_query_type(question)
        bm25_weight, vector_weight = get_dynamic_weights(query_type)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight]
        )
        ensemble_docs = ensemble_retriever.get_relevant_documents(question)

        # 计算每种方法的召回率和准确率
        ref_docs = reference_docs.get(question, [])

        def calculate_metrics(retrieved_docs):
            retrieved_texts = [doc.page_content for doc in retrieved_docs]

            # 准确率: 检索结果中相关文档的比例
            relevant_retrieved = sum(1 for doc in retrieved_texts if doc in ref_docs)
            precision = relevant_retrieved / len(retrieved_texts) if retrieved_texts else 0

            # 召回率: 检索到的相关文档占所有相关文档的比例
            recall = relevant_retrieved / len(ref_docs) if ref_docs else 0

            # 平均相似度
            similarities = []
            for doc in retrieved_texts:
                max_sim = max(calculate_semantic_similarity(doc, ref_doc) for ref_doc in ref_docs) if ref_docs else 0
                similarities.append(max_sim)
            avg_similarity = np.mean(similarities) if similarities else 0

            return precision, recall, avg_similarity

        bm25_precision, bm25_recall, bm25_sim = calculate_metrics(bm25_docs)
        vector_precision, vector_recall, vector_sim = calculate_metrics(vector_docs)
        ensemble_precision, ensemble_recall, ensemble_sim = calculate_metrics(ensemble_docs)

        results.append({
            "question": question,
            "query_type": query_type,
            "bm25_weight": bm25_weight,
            "vector_weight": vector_weight,
            "bm25_precision": bm25_precision,
            "bm25_recall": bm25_recall,
            "bm25_avg_sim": bm25_sim,
            "vector_precision": vector_precision,
            "vector_recall": vector_recall,
            "vector_avg_sim": vector_sim,
            "ensemble_precision": ensemble_precision,
            "ensemble_recall": ensemble_recall,
            "ensemble_avg_sim": ensemble_sim
        })

    return pd.DataFrame(results)


def evaluate_answers(
        questions: List[str],
        reference_answers: Dict[str, str]
) -> pd.DataFrame:
    """
    评估回答质量
    """
    results = []

    for question in questions:
        ref_answer = reference_answers.get(question, "")

        # 获取系统回答
        response = get_qa_chain_with_memory(question, evaluate_mode=True)
        answer = response["answer"]

        # 计算指标
        similarity = calculate_semantic_similarity(answer, ref_answer)

        # 评估回答长度
        answer_length = len(answer.split())
        ref_length = len(ref_answer.split())
        length_ratio = answer_length / ref_length if ref_length > 0 else 1

        # 评估响应时间
        response_time = response["response_time"]

        results.append({
            "question": question,
            "query_type": response["query_type"],
            "similarity": similarity,
            "length_ratio": length_ratio,
            "response_time": response_time,
            "system_answer": answer,
            "reference_answer": ref_answer
        })

    return pd.DataFrame(results)


def evaluate_weight_allocation(questions: List[str]) -> pd.DataFrame:
    """
    评估权重分配策略
    """
    results = []

    for question in questions:
        query_type = determine_query_type(question)
        dynamic_weights = get_dynamic_weights(query_type)

        # 测试不同权重组合
        weight_combinations = [
            (0.7, 0.3),  # 偏BM25
            (0.3, 0.7),  # 偏向量
            (0.5, 0.5),  # 平衡
            dynamic_weights  # 动态分配
        ]

        best_score = -1
        best_weights = (0, 0)

        for weights in weight_combinations:
            # 模拟检索过程
            vectordb = get_vectordb()
            documents = get_text_list_from_milvus(collection_name="Vmaxs")

            bm25_retriever = BM25Retriever.from_texts(documents)
            vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=weights
            )

            docs = ensemble_retriever.get_relevant_documents(question)

            # 简单评分: 检索结果的平均长度(假设更长的文档包含更多信息)
            score = np.mean([len(doc.page_content) for doc in docs]) if docs else 0

            if score > best_score:
                best_score = score
                best_weights = weights

        results.append({
            "question": question,
            "query_type": query_type,
            "dynamic_weights": dynamic_weights,
            "best_weights": best_weights,
            "weight_match": dynamic_weights == best_weights,
            "best_score": best_score
        })

    return pd.DataFrame(results)


def generate_evaluation_report(
        questions: List[str],
        reference_data: Dict[str, Dict]
) -> None:
    """
    生成完整的评估报告
    """
    print("=" * 50)
    print("混合检索问答系统评估报告")
    print("=" * 50)

    # 准备参考数据
    reference_docs = {q: data["docs"] for q, data in reference_data.items()}
    reference_answers = {q: data["answer"] for q, data in reference_data.items()}

    # 1. 检索效果评估
    print("\n1. 检索效果评估:")
    retrieval_df = evaluate_retrieval(questions, reference_docs)
    print(retrieval_df)

    # 计算平均指标
    avg_metrics = {
        "bm25_precision": retrieval_df["bm25_precision"].mean(),
        "bm25_recall": retrieval_df["bm25_recall"].mean(),
        "vector_precision": retrieval_df["vector_precision"].mean(),
        "vector_recall": retrieval_df["vector_recall"].mean(),
        "ensemble_precision": retrieval_df["ensemble_precision"].mean(),
        "ensemble_recall": retrieval_df["ensemble_recall"].mean()
    }
    print("\n平均检索指标:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.2f}")

    # 2. 问答质量评估
    print("\n2. 问答质量评估:")
    answer_df = evaluate_answers(questions, reference_answers)
    print(answer_df)

    avg_similarity = answer_df["similarity"].mean()
    avg_response_time = answer_df["response_time"].mean()
    print(f"\n平均回答相似度: {avg_similarity:.2f}")
    print(f"平均响应时间: {avg_response_time:.2f}秒")

    # 3. 权重分配评估
    print("\n3. 权重分配评估:")
    weight_df = evaluate_weight_allocation(questions)
    print(weight_df)

    weight_accuracy = weight_df["weight_match"].mean()
    print(f"\n权重分配准确率: {weight_accuracy:.2%}")

    # 保存评估结果
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    retrieval_df.to_csv(f"retrieval_evaluation_{timestamp}.csv", index=False)
    answer_df.to_csv(f"answer_evaluation_{timestamp}.csv", index=False)
    weight_df.to_csv(f"weight_evaluation_{timestamp}.csv", index=False)

    print("\n评估完成! 结果已保存到CSV文件。")


# ================== 测试和评估 ==================

if __name__ == "__main__":
    # 测试问题和参考数据
    test_questions = [
        "什么是VMAX的上网日志业务？",
        "上网日志业务包含哪些功能？",
        "为什么我的VMAX设备会出现日志丢失问题？",
        "如何解决VMAX日志存储空间不足的问题？",
        "VMAX-S与其他型号的主要区别是什么？"
    ]

    # 示例参考数据(实际使用时需要准备真实数据)
    reference_data = {
        "什么是VMAX的上网日志业务？": {
            "docs": [
                "VMAX上网日志业务是指记录用户上网行为的服务",
                "该业务可以记录用户访问的网站、时间和流量等信息"
            ],
            "answer": "VMAX上网日志业务是专门用于记录和存储用户上网行为的系统功能..."
        },
        "上网日志业务包含哪些功能？": {
            "docs": [
                "功能包括:1.访问记录 2.流量统计 3.行为分析",
                "主要功能模块:日志采集、存储、分析和报表"
            ],
            "answer": "上网日志业务主要包含以下功能:\n1. 访问记录\n2. 流量统计\n3. 行为分析\n4. 报表生成"
        },
        # 其他问题的参考数据...
    }

    # 运行评估
    generate_evaluation_report(test_questions, reference_data)

    # 交互式问答演示
    print("\n交互式问答演示(输入'退出'结束):")
    while True:
        question = input("\n请输入问题: ")
        if question.lower() == '退出':
            break

        result = get_qa_chain_with_memory(question)
        print(f"\n回答: {result['answer']}")