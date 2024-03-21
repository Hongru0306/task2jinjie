from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding
import os

# 没有保存数据库

os.chdir('/CV/xhr_project/llm/Paper/constructionGPT/rag-learning-master')
embedding = ZhipuEmbedding() # 创建EmbeddingModel
if bool(os.listdir('./storage')):
    vector = VectorStore()
    vector.load_vector('./storage') # 加载本地的数据库
else:
    docs = ReadFiles('/CV/xhr_project/llm/Paper/constructionGPT/dataset').get_content(max_token_len=600, cover_content=150) # 获得data目录下的所有文件内容并分割
    vector = VectorStore(docs)
    vector.get_vector(EmbeddingModel=embedding)
    vector.persist(path='./storage') # 将向量和文档内容保存到storage目录下，下次再用就可以直接加载本地的数据库



messages = []
chat = InternLMChat('/CV/xhr_project/llm/model/internlm2-chat-7b/Shanghai_AI_Laboratory/internlm2-chat-7b')
print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    if input_text == "exit":
        break
    elif '请用原版模型,' in input_text:
        input_text = input_text.replace('请用原版模型,', '')
        content = ''
        response = chat.chat(input_text, [], content, ori_model=True)
    else:
        content = vector.query(input_text, EmbeddingModel=embedding, k=1)[0]
        response = chat.chat(input_text, [], content)
    messages.append((content, response))
    print(f"robot >>> {response}")





# # 保存数据库之后


# question = '逆向纠错的原理是什么？'

# content = vector.query(question, EmbeddingModel=embedding, k=1)[0]
# # chat = OpenAIChat(model='gpt-3.5-turbo-1106')
# chat = InternLMChat('/CV/xhr_project/llm/model/internlm/internlm-chat-7b')
# print(chat.chat(question, [], content))

