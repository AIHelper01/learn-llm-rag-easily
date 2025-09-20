# 🚀 Learn LLM RAG Easily

本项目旨在帮助开发者快速上手 **检索增强生成 (RAG, Retrieval-Augmented Generation)** 技术，从理论到实战，逐步搭建属于自己的知识库问答应用。

<p align="center">
  <img src="./assets/RAG封面.png" alt="RAG Cover" width="600">
</p>

---

## 📚 项目内容

项目分为三个核心部分：

### 1. RAG 理论基础
- [00_RAG入门](./01_RAG理论/00_RAG入门.md)
- [01_RAG进阶](./01_RAG理论/01_RAG进阶.md)
- [02_现有RAG框架总结](./01_RAG理论/02_现有RAG框架总结.md)
- [RAG 课程：吴恩达 - LangChain](./01_RAG理论/03_RAG-吴恩达-LangChain.md)

### 2. RAG 实战 (LangChain)
- [服务器准备](./02_RAG实战-LangChain/01_服务器准备/readme.md)
- [模型准备 (LLM, Embedding, Reranker)](./02_RAG实战-LangChain/02_模型准备)
- [数据准备与清洗](./02_RAG实战-LangChain/03_数据准备)
- [知识库搭建 (Chroma / Milvus)](./02_RAG实战-LangChain/04_知识库搭建)
- [RAG 应用开发](./02_RAG实战-LangChain/05_构建RAG应用)
- [进阶技巧：混合检索、Rerank、分块优化](./02_RAG实战-LangChain/06_RAG进阶技巧)
- [评估与优化](./02_RAG实战-LangChain/RECYCLER/07_系统评估与优化)

### 3. 配套环境
- `docker-compose.yml` → 启动 Milvus + MinIO + etcd
- `requirements.txt` → Python 依赖列表
- `volumes/` → 容器挂载存储目录

---

## ⚡ 快速开始

### 1. 克隆项目
```bash
git clone https://gitee.com/coderwillyan/learn-llm-rag-easily.git
cd learn-llm-rag-easily
````

### 2. 安装依赖

```bash
conda create -n env_rag python=3.10 -y
conda activate env_rag
pip install ipykernel  
python -m ipykernel install --user --name=env_rag --display-name "Python 3 (env_rag)"

pip install -r requirements.txt
```

---

## 🗂️ 项目结构

```bash
.
├── 01_RAG理论              # RAG 理论与原理
├── 02_RAG实战-LangChain    # 实战项目：LangChain + Milvus/Chroma
├── assets                  # 插图与配图
├── requirements.txt        # Python 依赖
└── README.md               # 项目说明
```

---

## 📖 学习路径建议

1. **理论入门** → 先学习 [01\_RAG理论](./01_RAG理论) 下的文档与笔记
2. **基础实战** → 完成 [02\_RAG实战-LangChain](./02_RAG实战-LangChain) 的 Demo
3. **搭建知识库** → 使用 Chroma 或 Milvus 构建自己的向量数据库
4. **优化应用** → 尝试混合检索、Rerank、Prompt Engineering 等进阶技巧
5. **评估与迭代** → 使用效果评估方法，不断改进 RAG 系统

---

## 🛠️ 技术栈

* **大模型框架**: LangChain
* **向量数据库**: Milvus, Chroma
* **部署工具**: Docker, Streamlit
* **评估优化**: AI Eval, chunking, rerank

---

## 🌟 适合人群

* 想要学习 **RAG 原理与最佳实践** 的开发者
* 想要快速搭建 **企业知识库问答应用** 的工程师
* 想要了解 **LLM 与检索结合落地** 的研究人员


---

## 🙏 致谢

* [LangChain](https://www.langchain.com/)
* [Milvus](https://milvus.io/)
* [Chroma](https://www.trychroma.com/)
* 吴恩达 RAG 课程

---

🔥 **学习 RAG，从这里开始！**


