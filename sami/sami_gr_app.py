import gradio as gr
import asyncio
from langchain.chat_models import ChatOpenAI
#使用 异步的 Callback   AsyncIteratorCallbackHandler
from langchain.callbacks import AsyncIteratorCallbackHandler

async def f():
   callback = AsyncIteratorCallbackHandler()
   llm = ChatOpenAI(engine='GPT-35',streaming=True,callbacks=[callback])
   coro = llm.apredict("写一个1000字的修仙小说")  # 这里如果是 LLMChain的话 可以 换成  chain.acall()
   asyncio.create_task(coro)
   text = ""
   async for token in callback.aiter():
       text = text+token
       yield gr.TextArea.update(value=text)

with gr.Blocks() as demo:
    with gr.Column():
         摘要汇总 = gr.TextArea(value="",label="摘要总结",)
         bn = gr.Button("触发", variant="primary")
    bn.click(f,[],[摘要汇总])

demo.queue().launch(share=False, inbrowser=False, server_name="0.0.0.0", server_port=8001)
