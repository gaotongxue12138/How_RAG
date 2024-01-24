from generator import *
from retriever import *

data = knowledge_store("./data/")
test_ = test_data("./test_data/")
retriever = retriever_model('./model/retriever/')

# use retriever 将测试数据和外接的知识库存储到向量知识库中。

#读取数据