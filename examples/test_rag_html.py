import os
import sys
import lazyllm
from lazyllm.tools.rag import Document
from lazyllm import SentenceSplitter, pipeline, parallel, Retriever, Reranker, bind, launchers, deploy

from bs4 import BeautifulSoup
from lazyllm.tools.rag import DocNode

# 获取当前文件的绝对路径
current_file = os.path.abspath(__file__)

# 获取当前文件的父级的父级目录（即向上两层）
deep_dir = os.path.abspath(os.path.join(current_file, '..', '..'))

# 添加该路径到 sys.path 中，确保 Python 可以找到深层模块
if deep_dir not in sys.path:
    sys.path.append(deep_dir)

# 可选：进一步添加整个项目根目录（再向上一级）
project_root = os.path.abspath(os.path.join(deep_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


# --- 自定义 HTML 处理器 ---
def processHtml(file, extra_info=None):
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()
    soup = BeautifulSoup(data, 'lxml')
    text = '\n'.join(s for s in soup.stripped_strings)
    return [DocNode(text=text, metadata=extra_info or {})]

# --- Prompt 定义 ---
prompt = (
    "You will play the role of an AI Q&A assistant and complete a dialogue task. "
    "In this task, you need to provide your answer based on the given context and question."
)

# --- 文档处理 ---
documents = Document(
    # dataset_path=os.path.join(os.getcwd(), "rag_data"),
    dataset_path="/home/baizy/project/LazyLLM/examples/webPage.html",
    # embed=lazyllm.OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"),
    # embed=lazyllm.TrainableModule('internlm2-chat-7b'),
    embed=lazyllm.TrainableModule('/home/baizy/.cache/modelscope/hub/models/Qwen/Qwen3-1.7B'),
    manager=False
)
# 句子级 node 分组
documents.create_node_group(
    name="sentences",
    transform=SentenceSplitter,
    chunk_size=1024,
    chunk_overlap=100
)
documents.add_reader("*.html", processHtml)
# 句子级 node 分组
documents.create_node_group(
    name="sentences",
    transform=SentenceSplitter,
    chunk_size=1024,
    chunk_overlap=100
)
# 初始化文档对象（替换为你的文档目录路径）
# documents = Document(dataset_path="/home/baizy/project/LazyLLM/examples/webPage.html")

# 注册 HTML 文件处理器
# documents.add_reader("*.html", processHtml)

# 加载指定的 HTML 文件（确保路径和文件名正确）
# data = documents._impl._reader.load_data(input_files=["webPage.html"])
# retriever = lazyllm.Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese",
#                               topk=3, output_format='content', join=' ') 
# retriever.start()

# --- 构建 Pipeline ---
with pipeline() as ppl:
    # with parallel() as prl:
        # prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        # prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)
        # prl.retriever2 = Retriever(documents, group_name="coarse_chunk", similarity="bm25_chinese", topk=3)
        # ppl.reranker = Reranker(
        #     "ModuleReranker",
        #     model=lazyllm.TrainableModule(type="rerank", embed_model_name="rerank"),
        #     topk=1,
        #     output_format='content',
        #     join=True
        # ) | bind(query=ppl.input)

    ppl.formatter = (
        lambda nodes, query: dict(context_str=nodes, query=query)
    ) | bind(query=ppl.input)

    # ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(
    ppl.llm = lazyllm.TrainableModule('/home/baizy/.cache/modelscope/hub/models/Qwen/Qwen3-8B').prompt(
        lazyllm.ChatPrompter(prompt, extra_keys=["context_str"])
    # ).deploy_method(deploy.vllm, launcher=launchers.remote(ngpus=1))
    ).deploy_method(deploy.vllm).start()
    # ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
    # ppl.sd3.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))


# lazyllm.WebModule(ppl, port=23466).start().wait()

query = input('HTML 文件中关于模型结构设计的内容是什么？')
res = ppl(query)
print(f'With RAG Answer: {res}')