#!/usr/bin/env python
# coding: utf-8

# embedding选型
from langchain_community.embeddings import OllamaEmbeddings
my_emb = OllamaEmbeddings(base_url='http://localhost:11434',model="bge-m3:latest")

# 批量处理文件夹中所有文件
import os

# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = '../data_base/knowledge_path/VMAX-S'
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_paths.append(file_path)
print(file_paths)

from langchain.document_loaders.pdf import PyMuPDFLoader

# from langchain.document_loaders.markdown import UnstructuredMarkdownLoader

# 遍历文件路径并把实例化的loader存放在loaders里
loaders = []

for file_path in file_paths:
    file_type = file_path.split('.')[-1]
    if file_type == 'pdf':
        loaders.append(PyMuPDFLoader(file_path))
    else:
        print(f"Unsupported file type: {file_type} for file {file_path}")

# 下载文件并存储到text
# 加载所有文档内容到 texts
texts = []
for loader in loaders:
    texts.extend(loader.load())  # 关键步骤：初始化 texts

# 作数据清洗
# 修改后的数据清洗部分（替换原始代码中对应段落）
import re

# 预编译正则表达式（提升效率）
linebreak_pattern = re.compile(
    r'(?<![\\u4e00-\\u9fff])\n(?![\\u4e00-\\u9fff])',  # 负向断言匹配非中文环境换行
    flags=re.DOTALL
)
space_pattern = re.compile(r'[ 　]+')  # 匹配半角/全角空格
special_chars = ['•', '▪', '▫', '▶', '®', '©']  # 可扩展的干扰符号列表

# 替换原始代码中的清洗循环
for text in texts:
    # 1. 清理非中文环境换行
    text.page_content = re.sub(
        linebreak_pattern,
        lambda m: m.group().replace('\n', ''),
        text.page_content
    )

    # 2. 批量清理特殊符号
    for char in special_chars:
        text.page_content = text.page_content.replace(char, '')

    # 3. 安全删除空格（保留URL等特殊场景）
    text.page_content = space_pattern.sub('', text.page_content)

''' 
* RecursiveCharacterTextSplitter 递归字符文本分割
RecursiveCharacterTextSplitter 将按不同的字符递归地分割(按照这个优先级["\n\n", "\n", " ", ""])，
    这样就能尽量把所有和语义相关的内容尽可能长时间地保留在同一位置
RecursiveCharacterTextSplitter需要关注的是4个参数：

* separators - 分隔符字符串数组
* chunk_size - 每个文档的字符数量限制
* chunk_overlap - 两份文档重叠区域的长度
* length_function - 长度计算函数
'''
# 导入文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 知识库中单段文本长度
CHUNK_SIZE = 512

# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)

split_docs = text_splitter.split_documents(texts)
print(f"切分后的文件数量：{len(split_docs)}")

print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")

split_docs[90].page_content

import os
from langchain_community.vectorstores import Milvus
from langchain_core.documents import Document

# 定义持久化目录
# 向量库创建
connection_args = {
    "host": "129.201.70.32",
    "port": "19530",
}


# 定义每批处理的文档数量
batch_size = 30

# 如果只想导入部分数据
# split_docs = split_docs[:3]
try:
    # 计算总批次数
    total_batches = (len(split_docs) + batch_size - 1) // batch_size
    
    # 初始化向量数据库（如果是第一次创建）
    vectordb = None
    
    for batch_num in range(total_batches):
        # 计算当前批次的起始和结束索引
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, len(split_docs))
        
        # 获取当前批次的文档
        batch_docs = split_docs[start_idx:end_idx]
        
        print(f"正在处理第 {batch_num + 1}/{total_batches} 批文档 (文档 {start_idx}-{end_idx-1})")

        if batch_num == 0:
            # 第一次创建向量数据库
            vectordb = Milvus.from_documents(
            documents=batch_docs,
            embedding=my_emb,
            collection_name="Vmaxs",
            drop_old=False,
            connection_args=connection_args,
            )

        else:
            # 后续批次添加到现有集合
            vectordb.add_documents(batch_docs)
        
        # 每批处理后持久化
        # vectordb.persist()
        print(f"第 {batch_num + 1} 批文档已成功导入并持久化")
    
    print("所有文档已成功导入并持久化到向量数据库。")
    
except Exception as e:
    print(f"处理过程中发生错误: {e}")
