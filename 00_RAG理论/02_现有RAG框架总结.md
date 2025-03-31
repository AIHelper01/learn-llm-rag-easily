## **十七个传统RAG框架**

传统的RAG框架，指的是集chunk切分、向量化、存储、检索、生成等几个阶段于一体的RAG框架，其核心在于其中的不同策略适应，如文档处理、检索策略等，代表性的如RAGFlow(深度文档理解)，也包括QAnything（重排rerank引入），也包括可高度配置的Dify等，大致雷同，

![传统rag](assets/传统rag.png)



下面归置17个：

1、AnythingLLM，具备完整的RAG（检索增强生成）和AI代理能力。地址：https://github.com/Mintplex-Labs/anything-llm

2、MaxKB，基于大型语言模型的知识库问答系统。即插即用，支持快速嵌入到第三方业务系统。地址：https://github.com/1Panel-dev/MaxKB

3、RAGFlow，一个基于**深度文档理解**的开源RAG（检索增强生成）引擎。地址：https://github.com/infiniflow/ragflow

4、Dify，一个开源的大型语言模型应用开发平台。Dify直观的界面结合了AI工作流、RAG流程、代理能力、模型管理、可观测性功能等，让您能快速从原型阶段过渡到生产阶段。地址：https://github.com/langgenius/dify

5、FastGPT，基于LLM构建的知识型平台，提供即开即用的数据加工和模型调用能力，允许通过流程可视化进行工作流编排。地址：https://github.com/labring/FastGPT

6、Langchain-Chatchat，基于Langchain和ChatGLM等不同大模型的本地知识库问答。地址：https://github.com/chatchat-space/Langchain-Chatchat

7、QAnything，基于Anything的问题和答案。地址：https://github.com/netease-youdao/QAnything

8、Quivr，使用Langchain、GPT 3.5/4 turbo、Private、Anthropic、VertexAI、Ollama、LLMs、Groq等与文档（PDF、CSV等）和应用程序交互，本地和私有的替代OpenAI GPTs和ChatGPT。地址：https://github.com/QuivrHQ/quivr

9、RAG-GPT，RAG-GPT利用LLM和RAG技术，从用户自定义的知识库中学习，为广泛的查询提供上下文相关的答案，确保快速准确的信息检索。地址：https://github.com/open-kf/rag-gpt

10、Verba，由Weaviate驱动的检索增强生成（RAG）聊天机器人。地址：https://github.com/weaviate/Verba

11、FlashRAG，一个用于高效RAG研究的Python工具包。地址：https://github.com/RUC-NLPIR/FlashRAG

12、LightRAG，检索器-代理-生成器式的RAG框架。地址：https://github.com/SylphAI-Inc/LightRAG

13、kotaemon，一个开源的干净且可定制的RAG UI。地址：https://github.com/Cinnamon/kotaemon，

14、RAGapp，在企业中使用Agentic RAG的最简单方式。地址：https://github.com/ragapp/ragapp

15、TurboRAG，通过预计算的KV缓存加速检索增强生成，适用于分块文本。地址：https://github.com/MooreThreads/TurboRAG

16、TEN，实时多模态AI代理框架。地址：https://github.com/TEN-framework/ten_framework

17、AutoRAG，RAG AutoML工具。地址：https://github.com/Marker-Inc-Korea/AutoRAG



## **七个GraphRAG框架**

GraphRAG框架这是流行于微软的GraphRAG，然后后续出现了很多轻量化的改进版本，如LightRAG、nano-GraphRAG，也有一些具有特色的版本，如KAG，其核心思想是在原先传统RAG的基础上，增加实体、社区、chunk之间的关联，或者原有KG的知识，从而提升召回和准确性。

这里总结7个：

1、LightRAG，简单快速的Graphrag检索增强生成。地址：https://github.com/HKUDS/LightRAG

2、GraphRAG-Ollama-UI，使用Ollama的GraphRAG，带有Gradio UI和额外功能。地址：https://github.com/severian42/GraphRAG-Ollama-UI

3、microsoft-GraphRAG，一个模块化的基于图的检索增强生成（RAG）系统。地址：https://github.com/microsoft/graphrag

4、nano-GraphRAG，一个简单、易于修改的GraphRAG实现。地址：https://github.com/gusye1234/nano-graphrag

5、KAG，基于OpenSPG引擎的知识增强生成框架，用于构建知识增强的严格决策制定和信息检索知识服务。地址：https://github.com/OpenSPG/KAG

6、Fast-GraphRAG，GraphRAG的轻量化版本。地址：https://github.com/circlemind-ai/fast-graphrag

7、Tiny-GraphRAG，一个小巧的GraphRAG实现。地址：https://github.com/limafang/tiny-graphrag