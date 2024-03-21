# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
import streamlit as st
from RAG.VectorBase import VectorStore
from RAG.utils import ReadFiles
from RAG.LLM import OpenAIChat, InternLMChat
from RAG.Embeddings import JinaEmbedding, ZhipuEmbedding
from modelscope import snapshot_download
import os


embedding = ZhipuEmbedding() # 创建EmbeddingModel
if bool(os.listdir('./storage')):
    vector = VectorStore()
    vector.load_vector('./storage') # 加载本地的数据库


# 在侧边栏中创建一个标题和一个链接
with st.sidebar:
    st.markdown("## Construction chat")
    "[建筑施工GPT Construction-chat]"
    # 创建一个滑块，用于选择最大长度，范围在0到1024之间，默认值为512
    max_length = st.slider("max_length", 0, 1024, 512, step=1)

# 创建一个标题和一个副标题
st.title("💬 Construction Chatbot")
st.caption("🚀 A streamlit chatbot powered by internLM2")

# 定义模型路径
model_id = 'Shanghai_AI_Laboratory/internlm2-math-7b'

model_name_or_path = snapshot_download(model_id, revision='master')

@st.cache_resource
def load_model(mode_name_or_path):
    return InternLMChat(mode_name_or_path)

chat = load_model(mode_name_or_path)

def prompt_process(input_text):
    if '请用原版模型,' in input_text:
        input_text = input_text.replace('请用原版模型,', '')
        content = ''
        mode = True
    else:
        content = vector.query(input_text, EmbeddingModel=embedding, k=1)[0]
        mode = False
    return input_text, content, mode

# 如果session_state中没有"messages"，则创建一个包含默认消息的列表
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "有什么可以帮您的？"}]

# 遍历session_state中的所有消息，并显示在聊天界面上
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# 如果用户在聊天输入框中输入了内容，则执行以下操作
if prompt := st.chat_input():
    input_text, content, mode = prompt_process(prompt)
    print(input_text, content, mode)
    response = chat.chat(input_text, [], content, ori_model=mode)

    # 将用户的输入添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 在聊天界面上显示用户的输入
    st.chat_message("user").write(prompt)
    
    # 将模型的输出添加到session_state中的messages列表中
    st.session_state.messages.append({"role": "assistant", "content": response})
    # 在聊天界面上显示模型的输出
    st.chat_message("assistant").write(response)
