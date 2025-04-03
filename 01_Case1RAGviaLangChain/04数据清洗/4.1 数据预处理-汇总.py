import os
# from dotenv import load_dotenv, find_dotenv 
# pip install python-dotenv

# _ = load_dotenv(find_dotenv())


# 获取folder_path下所有文件路径，储存在file_paths里
file_paths = []
folder_path = '../data_base/knowledge_path'
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
# 作数据清洗
from langchain.document_loaders.pdf import PyMuPDFLoader
import re
# 下载文件并存储到text
texts = []
for loader in loaders:
    texts.extend(loader.load())   

for text in texts:
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    text.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), text.page_content)
    text.page_content = text.page_content.replace('•', '')
    text.page_content = text.page_content.replace(' ', '')



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
#导入文本分割器
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 知识库中单段文本长度
CHUNK_SIZE = 512
# 知识库中相邻文本重合长度
OVERLAP_SIZE = 50

# 使用递归字符文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP_SIZE
)
# text_splitter.split_text(text.page_content[0:1000])
split_docs = text_splitter.split_documents(texts)

print(f"---\n文档split成功")


print(f"切分后的文件数量：{len(split_docs)}")

print(f"切分后的字符数（可以用来大致评估 token 数）：{sum([len(doc.page_content) for doc in split_docs])}")


split_docs[2].page_content