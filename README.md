# 轻松学大模型应用开发

<div align=center>
<img src="figures/C0-0-logo.png" width = "1000">
</div>


## 项目概览

**本项目是基于langchain的大模型RAG外挂知识库的开发教程，主要内容包括：**

1. **大模型简介**，什么是大模型、大模型特点是什么、LangChain 是什么，如何实现RAG；
2. **如何调用大模型 API**，介绍了国内外知名大模型产品 （ChatGPT、智谱AI等）API 的多种调用方式，包括调用原生 API、封装为 LangChain LLM等调用方式；
3. **如何调用本地部署的大模型**，使用ollama完成大模型的本地化部署，调用本地大模型的能力；
4. **知识库搭建**，不同类型知识库文档的加载、处理，向量数据库的搭建；
5. **构建 RAG 应用**，包括将 LLM 接入到 LangChain 构建检索问答链，使用 Streamlit 进行应用部署
6. **验证迭代**，大模型开发如何实现验证迭代，一般的评估方法有什么；

**如果希望0代码完成本地知识库的搭建，可以借助一些优秀的开源项目：** 

1. [ragflow，一款基于深度文档理解构建的开源 RAG（Retrieval-Augmented Generation）引擎](https://github.com/infiniflow/ragflow/tree/main)
2. [AnythingLLM，一个全栈应用程序，您可以使用现成的商业大语言模型或流行的开源大语言模型，再结合向量数据库解决方案构建一个私有ChatGPT](https://github.com/Mintplex-Labs/anything-llm)
3. [MaxKB = Max Knowledge Base，是一款基于大语言模型和 RAG 的开源知识库问答系统，广泛应用于智能客服、企业内部知识库、学术研究与教育等场景。](https://github.com/1Panel-dev/MaxKB)



## 目录结构说明

```shell
data_base：知识库源文件和向量数据库文件
notebook：Notebook 源代码文件
requirements.txt：安装依赖
figures：图片
```



## 快速开始

1. 克隆仓库：

```
$ git clone https://gitee.com/coderwillyan/llm-rag
```

2. 安装依赖： 

进入项目llm-rag的根目录，执行pip install指令：

```
pip install -r requirements.txt
```

3. 使用jupyter lab依次执行代码



> 如果chromadb在windows上安装失败，参考：[issues: chromadb fails to install on windows](https://github.com/chroma-core/chroma/issues/189)



## 内容大纲

1. [LLM 介绍](./notebook/C1%20大型语言模型%20LLM%20介绍/) 
   1. [LLM 的理论介绍](./notebook/C1%20大型语言模型%20LLM%20介绍/1.大型语言模型%20LLM%20理论简介.md)
   2. [检索增强生成 RAG 简介](./notebook/C1%20大型语言模型%20LLM%20介绍/2.检索增强生成%20RAG%20简介.md)
   3. [什么是 LangChain](./notebook/C1%20大型语言模型%20LLM%20介绍/3.LangChain%20简介.md)
   4. [开发 LLM 应用的整体流程](./notebook/C1%20大型语言模型%20LLM%20介绍/4.开发%20LLM%20应用的整体流程.md)
   5. [AutoDL服务器的基本使用](./notebook/C1%20大型语言模型%20LLM%20介绍/5.AutoDL服务器的基本使用.md)
   7. [环境配置](./notebook/C1%20大型语言模型%20LLM%20介绍/7.环境配置.md)
2. [调用 LLM 实现基础问答](./notebook/C2%20调用%20LLM%20实现基础问答/) 
   1. [基本概念](./notebook/C2%20调用%20LLM%20实现基础问答/2.1%20基本概念.md)
   2. [使用 LLM API](./notebook/C2%20调用%20LLM%20实现基础问答/2.2%20使用%20LLM%20API.ipynb)
   3. [Prompt Engineering](./notebook/C2%20调用%20LLM%20实现基础问答/2.3%20Prompt%20Engineering.ipynb)
3. [构建词向量](./notebook/C3%20构建词向量%20WordEmbedding/) 
   1. [词向量及向量知识库介绍](./notebook/C3%20构建词向量%20WordEmbedding/3.1%20词向量及向量知识库介绍.md)
   2. [使用 Embedding API](./notebook/C3%20构建词向量%20WordEmbedding/3.2%20使用%20Embedding%20API.ipynb)
4. [构建词向量](./notebook/C4%20准备数据/) 
   1. [数据预处理处理：读取、清洗与切片](./notebook/C4%20准备数据/4.1%20数据预处理.ipynb)
5. [搭建向量数据库](./notebook/C5%20搭建知识库/) 
   1. [搭建向量数据库](/notebook/C5%20搭建知识库/5.1%20搭建向量数据库.ipynb)
6. [构建 RAG 应用](./notebook/C6%20构建%20RAG%20应用/) 
   1. [将 LLM 接入 LangChain](./notebook/C6%20构建%20RAG%20应用/6.1%20基于%20LangChain%20调用%20LLM.ipynb)
   2. [基于 LangChain 搭建检索问答链](./notebook/C6%20构建%20RAG%20应用/6.2%20构建检索问答链.ipynb)
   3. [基于 Streamlit 部署知识库助手](./notebook/C6%20构建%20RAG%20应用/6.3%20部署知识库助手.ipynb)
5. [系统评估与优化](./notebook/C5%20系统评估与优化/) 
   1. [如何评估 LLM 应用](./notebook/C7%20系统评估与优化/7.1%20如何评估%20LLM%20应用.ipynb)
   2. [评估并优化生成部分](./notebook/C7%20系统评估与优化/7.2%20评估并优化生成部分.ipynb)
   3. [评估并优化检索部分](./notebook/C7%20系统评估与优化/7.3%20评估并优化检索部分.md)



## 附RAG

![img](figures/rag1.png)



