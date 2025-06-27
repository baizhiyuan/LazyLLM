import os
import sys

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


from lazyllm.tools.rag import DocNode, Document
from bs4 import BeautifulSoup


def processHtml(file, extra_info=None):
    text = ''
    
    # 读取本地 HTML 文件
    with open(file, 'r', encoding='utf-8') as f:
        data = f.read()

    # 使用 BeautifulSoup 解析 HTML
    soup = BeautifulSoup(data, 'lxml')
    
    # 提取所有纯文本内容
    for element in soup.stripped_strings:
        text += element + '\n'

    # 封装为 DocNode 节点，附加元信息
    node = DocNode(text=text, metadata=extra_info or {})
    return [node]


# 初始化文档对象（替换为你的文档目录路径）
doc = Document(dataset_path="/home/baizy/project/LazyLLM/examples/webPage.html")

# 注册 HTML 文件处理器
doc.add_reader("*.html", processHtml)

# 加载指定的 HTML 文件（确保路径和文件名正确）
data = doc._impl._reader.load_data(input_files=["webPage.html"])

# 输出处理结果
print(f"data: {data}")
print(f"text: {data[0].text}")