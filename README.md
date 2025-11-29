# Learn LLM RAG Easily - 大模型轻松RAG指南

一个系统化的 **大模型检索增强生成（RAG）学习项目**，涵盖了从 **RAG 理论基础** 到 **主流RAG框架（LangChain / RAGFlow）** 的完整实践流程。

## 📁 项目结构

```
learn-llm-rag-easily/
│
├── assets/                    # 资源文件
├── 01_RAG理论/                # 01 - RAG Theory
├── 02_RAG实战-LangChain/      # 02 - Hands-on: LangChain
├── 03_RAG实战-RAGFlow/        # 03 - Hands-on: RAGFlow
├── main.py                   # 项目主入口文件
├── requirements.txt          # Python依赖包列表
├── pyproject.toml            # 项目配置
├── uv.lock                   # 依赖锁定文件
└── README.md                 # 项目说明
```

## 📚 学习线路

### 第一阶段：理论基础

1. **【01_RAG理论】**：理解检索增强生成的基本原理和核心组件

### 第二阶段：RAG框架实战

1. **【02_RAG实战-LangChain】**：使用LangChain构建灵活的RAG系统
2. **【03_RAG实战-RAGFlow】**：掌握RAGFlow这一开箱即用的RAG平台

## 🚀 快速开始

### 1. 克隆项目

```
git clone https://gitee.com/your-username/learn-llm-rag-easily.git
cd learn-llm-rag-easily
```

### 2. 创建并激活conda环境

推荐使用 **Python 3.10+**：

**推荐在提供的Jupyter/VSCode环境运行**：

```
conda create -n env_rag python=3.12 -y
conda activate env_rag

pip install ipykernel
python -m ipykernel install --user --name=env_vllm --display-name "Python3 (env_vllm)"
```

### 3. 安装依赖

```
pip install -r requirements.txt
```

### 4. 按目录学习

按照学习线路的顺序，进入各个目录查看详细的教程和代码：

```
cd 01_RAG理论/
# 阅读该目录下的文档和实践代码
```

