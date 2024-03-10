import pandas as pd
import torch
from datasets import load_dataset,Dataset
from torch.utils.data import Dataset
import pandas as pd



#nl,code,relevant_code
class RAG_Dataset(Dataset):
    def __init__(self, data_path, graph_path, block_size,generator_tokenizer,mode,code_tokens):
        

        self.generator_tokenizer = generator_tokenizer
        self.block_size = block_size
        self.mode = mode
        self.code_tokens = code_tokens
        
        self.inputs = []
        self.token_labels = []
        self.code_prompts = []

        #读取数据
        x = pd.read_json(data_path, lines= True)
        graph_list = torch.load(graph_path)
        
        for i in range(len(x)):                     
            code = generator_tokenizer.encode(x["code"][i])
            nl = generator_tokenizer.encode(x["nl"][i])
            code_graph = graph_list[i]

            input_ids, input_labels  = self.pad_and_get_mask(code, nl,generator_tokenizer)
            self.inputs.append(input_ids)
            self.token_labels.append(input_labels)
            self.code_prompts.append(code_graph)

    def pad_and_get_mask(self, code, nl , tokenizer):

        while (len(code) + len(nl) + 2 + self.code_tokens > self.block_size):
            if (len(code) > len(nl)):
                code = code[:-1]
            else:
                nl = nl[:-1]
        
        if self.mode == 'train':
            inputs = nl + [tokenizer.unk_token_id] * self.code_tokens + [tokenizer.bos_token_id] + code + [tokenizer.eos_token_id]
            labels = [1] * len(nl) + [1] * self.code_tokens + [2] * (len(code)+1) + [0]
        else:
            inputs = nl + [tokenizer.unk_token_id] * self.code_tokens + [tokenizer.bos_token_id]
            labels = [1] * len(nl) + [1] * self.code_tokens   + [2]
            return inputs, labels
        

        assert len(inputs) <= self.block_size
        pad_len = self.block_size - len(inputs)
        inputs += [tokenizer.unk_token_id] * pad_len
        labels += [0] * pad_len
        assert len(inputs) == len(labels)
    
        return inputs , labels


    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        
       
        return torch.tensor(self.inputs[item]), torch.tensor(self.token_labels[item]) ,  self.code_prompts[item]

            