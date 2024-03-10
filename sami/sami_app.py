import openai
import streamlit as st
from streamlit_chat import message
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools.retriever import create_retriever_tool
from langchain.callbacks import AsyncIteratorCallbackHandler
os.environ["OPENAI_API_KEY"] = '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'

#openai.api_key = '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'
embeddings = OpenAIEmbeddings()
db = FAISS.load_local("./test_218/", embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})


prompt = hub.pull("hwchase17/openai-tools-agent")
tool = create_retriever_tool(
    retriever,
    "search",
    "当用户提问义乌和商品时，需要调用此工具查询相关信息",
)
tools = [tool]

llm = ChatOpenAI(temperature=0)

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
def generate_response(prompt):

    result = agent_executor.invoke({"input": prompt})

    return result['output']


st.markdown("我是口语翻译专家 + 外贸能力很强的一个人")
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
user_input = st.text_input("请输入您的问题:", key='input')
if user_input:
    output = generate_response(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["gen erated"][i], key=str(i))
        message(st.session_state['past'][i],
                is_user=True,
                key=str(i) + '_user')