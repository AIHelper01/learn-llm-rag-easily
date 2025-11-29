# Milvus 架构概述

Milvus 构建在 Faiss、HNSW、DiskANN、SCANN 等流行的向量搜索库之上，专为在包含数百万、数十亿甚至数万亿向量的密集向量数据集上进行相似性搜索而设计。在继续之前，请先熟悉一下 Embeddings 检索的[基本原理](https://milvus.io/docs/zh/glossary.md)。

Milvus 还支持数据分片、流式数据摄取、动态 Schema、结合向量和标量数据的搜索、多向量和混合搜索、稀疏向量和其他许多高级功能。该平台可按需提供性能，并可进行优化，以适应任何嵌入式检索场景。我们建议使用 Kubernetes 部署 Milvus，以获得最佳的可用性和弹性。

Milvus 采用共享存储架构，其计算节点具有存储和计算分解及横向扩展能力。按照数据平面和控制平面分解的原则，Milvus 由[四层](https://milvus.io/docs/zh/four_layers.md)组成：访问层、协调器服务、工作节点和存储。这些层在扩展或灾难恢复时相互独立。

![Architecture_diagram](https://milvus.io/docs/v2.5.x/assets/milvus_architecture.png)架构图

根据该图，接口可分为以下几类：

- **DDL / DCL：**createCollection / createPartition / dropCollection / dropPartition / hasCollection / hasPartition
- **DML / Produce：**插入 / 删除 / 上移
- **DQL:**搜索/查询





# Milvus 部署选项概述

Milvus 是一个高性能、可扩展的向量数据库。它支持各种规模的用例，从在 Jupyter Notebooks 中本地运行的演示到处理数百亿向量的大规模 Kubernetes 集群。目前，Milvus 有三种部署选项：Milvus Lite、Milvus Standalone 和 Milvus Distributed。

## Milvus Lite

Milvus Lite是一个 Python 库，可导入到您的应用程序中。作为 Milvus 的轻量级版本，它非常适合在 Jupyter 笔记本或资源有限的智能设备上运行快速原型。Milvus Lite 支持与 Milvus 其他部署相同的 API。与 Milvus Lite 交互的客户端代码也能与其他部署模式下的 Milvus 实例协同工作。

要将 Milvus Lite 集成到应用程序中，请运行`pip install pymilvus` 进行安装，并使用`MilvusClient("./demo.db")` 语句实例化一个带有本地文件的向量数据库，以持久化所有数据。更多详情，请参阅运行 Milvus Lite。



### 设置 Milvus Lite

```shell
pip install -U pymilvus
```

我们建议使用`pymilvus` 。由于`milvus-lite` 已包含在`pymilvus` 2.4.2 或更高版本中，因此可通过`pip install` 与`-U` 强制更新到最新版本，`milvus-lite` 会自动安装。

如果你想明确安装`milvus-lite` 软件包，或者你已经安装了旧版本的`milvus-lite` 并想更新它，可以使用`pip install -U milvus-lite` 。

### 连接 Milvus Lite

在`pymilvus` 中，指定一个本地文件名作为 MilvusClient 的 uri 参数将使用 Milvus Lite。

```python
from pymilvus import MilvusClient
client = MilvusClient("./milvus_demo.db")
```

运行上述代码段后，将在当前文件夹下生成名为**milvus_demo.db 的**数据库文件。

> ***注意：\***请注意，同样的 API 也适用于 Milvus Standalone、Milvus Distributed 和 Zilliz Cloud，唯一的区别是将本地文件名替换为远程服务器端点和凭据，例如`client = MilvusClient(uri="http://localhost:19530", token="username:password")` 。

### 示例

以下是如何使用 Milvus Lite 进行文本搜索的简单演示。还有更多使用 Milvus Lite 构建[RAG](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/build_RAG_with_milvus.ipynb)、[图像搜索](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/quickstart/image_search_with_milvus.ipynb)等应用程序的综合[示例](https://github.com/milvus-io/bootcamp/tree/master/bootcamp/tutorials)，以及在[LangChain](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/rag_with_milvus_and_langchain.ipynb)和[LlamaIndex](https://github.com/milvus-io/bootcamp/blob/master/bootcamp/tutorials/integration/rag_with_milvus_and_llamaindex.ipynb) 等流行 RAG 框架中使用 Milvus Lite 的[示例](https://github.com/milvus-io/bootcamp/tree/master/bootcamp/tutorials)！

```python
from pymilvus import MilvusClient
import numpy as np

client = MilvusClient("./milvus_demo.db")
client.create_collection(
    collection_name="demo_collection",
    dimension=384  # The vectors we will use in this demo has 384 dimensions
)

# Text strings to search from.
docs = [
    "Artificial intelligence was founded as an academic discipline in 1956.",
    "Alan Turing was the first person to conduct substantial research in AI.",
    "Born in Maida Vale, London, Turing was raised in southern England.",
]
# For illustration, here we use fake vectors with random numbers (384 dimension).

vectors = [[ np.random.uniform(-1, 1) for _ in range(384) ] for _ in range(len(docs)) ]
data = [ {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"} for i in range(len(vectors)) ]
res = client.insert(
    collection_name="demo_collection",
    data=data
)

# This will exclude any text in "history" subject despite close to the query vector.
res = client.search(
    collection_name="demo_collection",
    data=[vectors[0]],
    filter="subject == 'history'",
    limit=2,
    output_fields=["text", "subject"],
)
print(res)

# a query that retrieves all entities matching filter expressions.
res = client.query(
    collection_name="demo_collection",
    filter="subject == 'history'",
    output_fields=["text", "subject"],
)
print(res)

# delete
res = client.delete(
    collection_name="demo_collection",
    filter="subject == 'history'",
)
print(res)
```



## Milvus 单机版

Milvus Standalone 是单机服务器部署。Milvus Standalone 的所有组件都打包到一个[Docker 镜像](https://milvus.io/docs/install_standalone-docker.md)中，部署起来非常方便。如果你有生产工作负载，但又不想使用 Kubernetes，那么在内存充足的单机上运行 Milvus Standalone 是一个不错的选择。此外，Milvus Standalone 通过主从复制支持高可用性。

### 使用 Docker Compose 运行 Milvus (Linux)

本页说明如何使用 Docker Compose 在 Docker 中启动 Milvus 实例。

### 安装 Milvus

Milvus 在 Milvus 资源库中提供了 Docker Compose 配置文件。要使用 Docker Compose 安装 Milvus，只需运行

```shell
# Download the configuration file
$ wget https://bgithub.xyz/milvus-io/milvus/releases/download/v2.5.6/milvus-standalone-docker-compose.yml -O docker-compose.yml

# Start Milvus
$ sudo docker compose up -d

Creating milvus-etcd  ... done
Creating milvus-minio ... done
Creating milvus-standalone ... done
```

- 如果运行上述命令失败，请检查系统是否安装了 Docker Compose V1。如果是这种情况，建议你根据[本页](https://docs.docker.com/compose/)的说明迁移到 Docker Compose V2。
- 如果您在拉取镜像时遇到任何问题，请通过[community@zilliz.com](mailto:community@zilliz.com)联系我们，并提供有关问题的详细信息，我们将为您提供必要的支持。

启动 Milvus 后、

- 名为milvus-standalone、milvus-minio和milvus-etcd的容器启动。
  - **milvus-etcd**容器不向主机暴露任何端口，并将其数据映射到当前文件夹中的**volumes/etcd**。
  - **milvus-minio**容器使用默认身份验证凭据在本地为端口**9090**和**9091**提供服务，并将其数据映射到当前文件夹中的**volumes/minio**。
  - **Milvus-standalone**容器使用默认设置为本地**19530**端口提供服务，并将其数据映射到当前文件夹中的**volumes/milvus**。

你可以使用以下命令检查容器是否启动并运行：

```shell
$ sudo docker-compose ps

      Name                     Command                  State                            Ports
--------------------------------------------------------------------------------------------------------------------
milvus-etcd         etcd -advertise-client-url ...   Up             2379/tcp, 2380/tcp
milvus-minio        /usr/bin/docker-entrypoint ...   Up (healthy)   9000/tcp
milvus-standalone   /tini -- milvus run standalone   Up             0.0.0.0:19530->19530/tcp, 0.0.0.0:9091->9091/tcp
```

你还可以访问 Milvus WebUI，网址是`http://127.0.0.1:9091/webui/` ，了解有关 Milvus 实例的更多信息。详情请参阅[Milvus WebUI](https://milvus.io/docs/zh/milvus-webui.md)。

### 停止和删除 Milvus

您可以按以下步骤停止和删除此容器

```shell
# Stop Milvus
$ sudo docker compose down

# Delete service data
$ sudo rm -rf volumes
```



## 分布式 Milvus

Milvus Distributed 可部署在[Kubernetes](https://milvus.io/docs/install_cluster-milvusoperator.md)集群上。这种部署采用云原生架构，摄取负载和搜索查询分别由独立节点处理，允许关键组件冗余。它具有最高的可扩展性和可用性，并能灵活定制每个组件中分配的资源。Milvus Distributed 是在生产中运行大规模向量搜索系统的企业用户的首选。

## 为您的使用案例选择正确的部署方式

部署模式的选择通常取决于应用程序的开发阶段：

- **用于快速原型开发**

  如果您想快速构建原型或用于学习，如检索增强生成（RAG）演示、人工智能聊天机器人、多模态搜索，Milvus Lite 本身或 Milvus Lite 与 Milvus Standalone 的组合都很适合。您可以在笔记本中使用 Milvus Lite 进行快速原型开发，并探索各种方法，如 RAG 中的不同分块策略。您可能希望在小规模生产中部署用 Milvus Lite 构建的应用程序，为真正的用户提供服务，或在更大的数据集（例如超过几百万个向量）上验证想法。Milvus Standalone 是合适的选择。Milvus Lite 的应用逻辑仍可共享，因为所有 Milvus 部署都有相同的客户端应用程序接口。Milvus Lite 中存储的数据也可以通过命令行工具移植到 Milvus Standalone 中。

- **小规模生产部署**

  对于早期生产阶段，当项目仍在寻求产品与市场的契合，敏捷性比可扩展性更重要时，Milvus Standalone 是最佳选择。只要有足够的机器资源，它仍然可以扩展到 1 亿向量，同时对 DevOps 的要求也比维护 K8s 集群低得多。

- **大规模生产部署**

  当你的业务快速增长，数据规模超过单台服务器的容量时，是时候考虑 Milvus Distributed 了。你可以继续使用Milvus Standalone作为开发或暂存环境，并操作运行Milvus Distributed的K8s集群。这可以支持你处理数百亿个向量，还能根据你的特定工作负载（如高读取、低写入或高写入、低读取的情况）灵活调整节点大小。

- **边缘设备上的本地搜索**

  对于在边缘设备上通过私有或敏感信息进行搜索，您可以在设备上部署 Milvus Lite，而无需依赖基于云的服务来进行文本或图像搜索。这适用于专有文档搜索或设备上对象检测等情况。

Milvus 部署模式的选择取决于项目的阶段和规模。Milvus 为从快速原型开发到大规模企业部署的各种需求提供了灵活而强大的解决方案。

- **Milvus Lite**建议用于较小的数据集，多达几百万个向量。
- **Milvus Standalone**适用于中型数据集，可扩展至 1 亿向量。
- **Milvus Distributed 专为**大规模部署而设计，能够处理从一亿到数百亿向量的数据集。

![Select deployment option for your use case](https://milvus.io/docs/v2.5.x/assets/select-deployment-option.png)选择适合您使用情况的部署选项

## 功能比较

| 功能             | Milvus Lite                                                  | Milvus 单机版                                                | 分布式 Milvus                                                |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SDK / 客户端软件 | Python gRPC                                                  | Python Go Java Node.js C# RESTful                            | Python Java Go Node.js C# RESTful                            |
| 数据类型         | 密集向量 稀疏向量 二进制向量 布尔值 整数 浮点 VarChar 数组 JSON | 密集向量 稀疏向量 二进制向量 布尔型 整数 浮点型 VarChar 数组 JSON | 密集向量 稀疏向量 二进制向量 布尔值 整数 浮点 VarChar 数组 JSON |
| 搜索功能         | 向量搜索（ANN 搜索） 元数据过滤 范围搜索 标量查询 通过主键获取实体 混合搜索 | 向量搜索（ANN 搜索） 元数据过滤 范围搜索 标量查询 通过主键获取实体 混合搜索 | 向量搜索（ANN 搜索） 元数据过滤 范围搜索 标量查询 通过主键获取实体 混合搜索 |
| CRUD 操作符      | ✔️                                                            | ✔️                                                            | ✔️                                                            |
| 高级数据管理     | 不适用                                                       | 访问控制 分区 分区密钥                                       | 访问控制 分区 分区密钥 物理资源分组                          |
| 一致性级别       | 强                                                           | 强 有界稳定性 会话 最终                                      | 强 有界稳定性 会话 最终                                      |





# Attu（Milvus 图形用户界面）

https://github.com/zilliztech/attu

```
docker run -p 9092:3000 -e MILVUS_URL=129.201.70.35:19530 zilliz/attu:v2.6
```



# 用户指南

- [数据库](https://milvus.io/docs/zh/manage_databases.md)
- Collections
- Schema 和数据字段
- 插入和删除
- 索引
- 搜索和 Rerankers

> 参考https://milvus.io/docs/zh/manage_databases.md

