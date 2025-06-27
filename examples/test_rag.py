import lazyllm
from lazyllm import SentenceSplitter, pipeline, parallel, Retriever, Reranker, bind, launchers, deploy

# 文档加载
documents = lazyllm.Document(dataset_path="/home/baizy/project/LazyLLM/examples/data_kb")
# 检索组件定义
# 如果下游节点是大模型，使用join；如果是重排器，那么不用
# retriever = lazyllm.Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese",
#                               topk=3, output_format='content', join=' ') 
# retriever.start()


# 生成组件定义
prompt = 'You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into English drawing prompt words, and you can enrich and expand the prompt word content.'
# prompt = ('你是一位知识渊博且富有同理心的虚拟健身教练，专注于制定个性化的健身计划并为各个健身水平的用户提供指导。你的目标是支持、教育和激励用户安全和可持续地实现他们的健身和健康目标。以友好、专业和鼓励的语气回应用户的询问，确保你的建议是可操作的、基于证据的，并根据用户的需求量身定制。在你的回答中，确保遵循以下原则：同理心和鼓励：始终保持积极、非评判的语气，鼓励用户，无论他们的健身水平或面临的挑战如何。示例：‘你迈出了第一步，这太棒了！让我们一起制定一个适合你日程的计划。’个性化：利用用户提供的信息，例如他们的健身目标、当前活动水平、饮食偏好和任何限制，以量身定制回答。示例：对于一个希望减肥的初学者，建议他们可以遵循的可管理的锻炼和膳食建议。基于证据的建议：提供基于科学证据的推荐，以简单的术语解释你建议背后的理由。示例：解释为什么力量训练有助于减肥目标或水合作用在恢复中的作用。安全第一：在所有建议中强调正确的姿势、逐步进展和预防伤害。如果用户报告疼痛或不适，建议咨询医疗专业人员。示例：‘如果你是跑步新手，建议从走路和慢跑的混合开始，逐渐建立耐力。’可操作步骤：将建议分解为用户可以轻松遵循的简单、可操作的步骤或例程。示例：提供一个初学者的锻炼计划，包含明确的组数、次数和休息间隔。教育性见解：以易于理解和实施的方式分享关于健身、营养或健康的教育性提示。示例：解释宏观营养素在肌肉生长中的重要性或动态拉伸在锻炼前的好处。多样化沟通：有效回应各种用户需求，例如：锻炼指导：建议特定的锻炼以实现减肥、增肌或灵活性等目标。饮食和营养：提供膳食建议、份量指导以及均衡饮食的见解。动机和心态：提供动机支持，帮助用户保持一致和积极。如果你没有与用户查询匹配的上下文，可以礼貌地说明该查询无法很好地回答。如果查询与健身教练无关，可以礼貌地拒绝回答，并说明你的目的。')

# llm = lazyllm.OnlineChatModule(source="sensenova").prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
# llm = lazyllm.TrainableModule('qwen3:14b-q8_0').start()
# llm = lazyllm.TrainableModule('internlm2-chat-7b').start()
# llm = lazyllm.TrainableModule('Qwen/Qwen3-8B').start()
# llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

with lazyllm.pipeline() as ppl:
    # ppl.retriever = lazyllm.Retriever(doc=documents, group_name="CoarseChunk", similarity="bm25_chinese",
    #                                   topk=3, output_format='content', join=' ')
    ppl.formatter = (lambda nodes, query: dict(context_str = nodes, query = query)) | bind(query = ppl.input)
    # ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b') #.start()
    # ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b-4bits') #.start()
    # ppl.llm = lazyllm.TrainableModule('Qwen/Qwen3-4B-Base').start()
    # ppl.llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))
    # ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
    # ppl.sd3 = lazyllm.TrainableModule('tensorart/stable-diffusion-3.5-medium-turbo')
    # ppl.sd3.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=['context_str']))

    ppl.llm = lazyllm.TrainableModule('/home/baizy/.cache/modelscope/hub/models/Qwen/Qwen3-8B'
              ).deploy_method(deploy.vllm)
    ppl.llm.prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))


# lazyllm.WebModule(ppl, port=23466, history=[ppl]).start().wait()
lazyllm.WebModule(ppl, port=23466).start().wait()

# 推理, 将query和召回节点中的内容组成dict，作为大模型的输入
query = input('请输入您的问题\n')
res = ppl(query)

print(f'With RAG Answer: {res}')