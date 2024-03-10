import os
from itertools import chain
from dataclasses import dataclass, field
from torch_geometric.data import Batch
from typing import Dict, Optional, Sequence, List
import torch
import transformers
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
from model import *
from dataset_builder import *
from transformers import AutoTokenizer,GPT2LMHeadModel
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm






@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    node_features: int = field(default=0, metadata={"help": "The number of node features."})
    gnn_hidden_dim: int = field(default=0, metadata={"help": "Hidden dimension size for GNN."})
    gnn_out_dim: int = field(default=0, metadata={"help": "Output dimension size for GNN."})
    n_layers: int = field(default=0, metadata={"help": "Number of layers in the GNN."})
    gnn_type: str = field(default="", metadata={"help": "Type of GNN (e.g., GCN, GAT)."})

    pro_in_features: int = field(default=0, metadata={"help": "Size of each input sample features."})
    pro_hidden_features: int = field(default=0, metadata={"help": "Size of hidden layers features."})
    code_tokens: int = field(default=128, metadata={"help": "List of code tokens."})



@dataclass
class DataArguments:
    train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    eval_path: str = field(default=None, metadata={"help": "Path to the evalation data."})
    t_graph_path: str = field(default=None)
    e_graph_path: str = field(default=None)
    block_size: int = field(default = 512)
    mode : str = field(default = "train")



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    batch_size: int = field(default = 8)
    lr: float = field(default=0.0001)
    epoch : int = field(default = 3)
    output_dir : str = field(default= "/root/autodl-tmp/")

def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = GNNCodeGPTForCausalLM.from_pretrained(model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code = True)

    model.get_model().init_GNN(model_args.node_features, model_args.gnn_hidden_dim, model_args.gnn_out_dim , model_args.n_layers , model_args.gnn_type)
   
    model.get_model().init_proj(model_args.pro_in_features,model_args.pro_hidden_features,model_args.code_tokens)
    

    train_dataset = RAG_Dataset(data_args.train_path, data_args.t_graph_path, data_args.block_size, tokenizer,"train",model_args.code_tokens)
    eval_dataset = RAG_Dataset(data_args.eval_path, data_args.e_graph_path, data_args.block_size, tokenizer,"eval",model_args.code_tokens)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(chain(model.adapter.parameters() , model.proj.parameters() , model.transformer.parameters()), lr = training_args.lr)

    def collate_fn(batch):
        # 解包batch得到三种类型的数据
        input_ids_list, token_labels_list, code_prompts_list = zip(*batch)
        input_ids_batch = torch.stack(input_ids_list)
        token_labels_batch = torch.stack(token_labels_list)        
        code_prompts_batch = Batch.from_data_list(code_prompts_list)
        
        return input_ids_batch, token_labels_batch, code_prompts_batch

    train_dataloader = DataLoader(train_dataset, batch_size = training_args.batch_size , shuffle=True ,collate_fn = collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size = 1 , shuffle=True ,collate_fn = collate_fn)

    device = training_args.device
    for epoch in range(training_args.epoch):
        model.train()
        total_loss = 0.0

        for step, (batch ,token_labels ,graph_data) in tqdm(enumerate(train_dataloader)):
          
            inputs = batch.cuda()
            graph_data = graph_data.cuda()

            attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8).cuda() 
            nl_mask = torch.tensor(token_labels.clone().detach() == 1, dtype=torch.uint8).cuda() 
            loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8).cuda()
                 
            outputs = model(input_ids = inputs,attention_mask = attn_mask, nl_mask = nl_mask , graph_data = graph_data )    
            logits = outputs.logits

            labels = inputs
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()      
            flatten_shift_loss_mask = loss_mask[..., :-1].contiguous().view(-1)
            ids = torch.nonzero(flatten_shift_loss_mask).view(-1)
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1))[ids], shift_labels.view(-1)[ids])
            total_loss += loss.item()
            loss.backward()  
            optimizer.step()
            optimizer.zero_grad()

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch + 1}/{training_args.epoch}, Average Loss: {average_loss}')

        # model.eval()
        # anss =[]
        # for step, (batch ,token_labels ,graph_data) in tqdm(enumerate(eval_dataloader)):
              
        #     inputs = batch.cuda()
        #     graph_data = graph_data.cuda()

        #     attn_mask = torch.tensor(token_labels.clone().detach() != 0, dtype=torch.uint8).cuda() 
        #     nl_mask = torch.tensor(token_labels.clone().detach() == 1, dtype=torch.uint8).cuda() 
        #     loss_mask = torch.tensor(token_labels.clone().detach() == 2, dtype=torch.uint8).cuda()
                 
        #     ans = model.generate(input_ids = inputs,attention_mask = attn_mask, nl_mask = nl_mask , graph_data = graph_data )    
        #     ans = tokenizer.decode(ans[0], skip_special_tokens=True)
        #     anss.append(ans)

        
#需要写一下评估函数

    parent_dir = os.path.dirname(training_args.output_dir)

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    torch.save(model.state_dict(), training_args.output_dir)











# def evalution():
#     parser = transformers.HfArgumentParser(
#         (ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     model = GNNCodeGPTForCausalLM.from_pretrained(model_args.model_name_or_path)
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code = True)
#     model.get_model().init_GNN(model_args.node_features, model_args.gnn_hidden_dim, model_args.gnn_out_dim , model_args.n_layers , model_args.gnn_type)
   
#     model.get_model().init_proj(model_args.pro_in_features,model_args.pro_hidden_features,model_args.code_tokens)
#     text = "hello!"
#     # 处理输入文本
#     inpu= tokenizer(text, return_tensors='pt')


#     # 使用模型生成文本
#     # 注意：这里假设你的GraphGPT2ForCausalLM已经正确继承并实现了.generate()方法
#     generated_output = model.generate(input_ids =inpu.input_ids,d,num_beams=1,max_length = 100)

#     # 将生成的token IDs转换回文本
#     generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

   



