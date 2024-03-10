from transformers import GPT2Config
from transformers import GPT2Model
from torch import nn
from transformers import GPT2PreTrainedModel,GPT2LMHeadModel,GenerationMixin,GPT2Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import GPT2Model
from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, global_add_pool

class GNNCodeGPTConfig(GPT2Config):
    model_type = "GNNCodeGPT"

class GNNCodeGPTModel(GPT2Model):
    config_class = GNNCodeGPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
    def init_GNN(self,node_features,gnn_hidden_dim,gnn_output_dim,n_layers,gnn_type):
        #node_features,gnn_hidden_dim,gnn_output_dim,n_layers,gnn_type
        self.adapter = GNNEncoder(node_features, gnn_hidden_dim, gnn_output_dim, n_layers, gnn_type)
    def init_proj(self,in_features,hidden_features,code_tokens):
        print(in_features,hidden_features,self.config.n_embd,code_tokens)
        self.proj = CodeProjection(in_features,hidden_features,self.config.n_embd,code_tokens)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        nl_mask = None,
        graph_data=None,
    ):


        if past_key_values is None:
 
            soft_prompt = self.adapter(graph_data)
            soft_prompt = self.proj(soft_prompt)
            inputs_embeds = self.wte(input_ids)
            print(inputs_embeds.shape)
            for i in range(inputs_embeds.shape[0]):
                ones_indices = nl_mask[i].nonzero(as_tuple=True)[0]
                start_idx = ones_indices[-1 * self.code_tokens]
                inputs_embeds[i, start_idx:start_idx + self.code_tokens] = soft_prompt[i]

            return super().forward(
                input_ids = None,  # 确保不再使用原始的input_ids
                attention_mask = attention_mask,
                past_key_values = past_key_values,
                inputs_embeds = inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        else:
            return super().forward(
                input_ids = input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )





class GNNCodeGPTForCausalLM(GPT2LMHeadModel):
    config_class = GNNCodeGPTConfig

    def __init__(self, config):
        super().__init__(config)
        #print("GPT的配置文件",config)
        self.transformer= GNNCodeGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):

        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "graph_data": [kwargs.get("graph_data", None)],
                # "edge_index_reps": kwargs.get("edge_index_reps", None),
            }
        )
        return model_inputs
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        nl_mask = None,
        graph_data=None,
    ):
        transformer_outputs = self.transformer(
            input_ids = input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            nl_mask = nl_mask,
            graph_data = graph_data
            # 注意：处理graph_data的逻辑应该在GraphGPT2Model中实现
        )

        hidden_states = transformer_outputs[0]
        logits = self.lm_head(hidden_states)

        # 下面的损失计算与处理逻辑与你之前的GraphLlamaForCausalLM类似

        return CausalLMOutputWithPast(logits=logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)
    def get_model(self):
        return self.transformer




class GNNEncoder(torch.nn.Module):
    r"""Graph Neural Networks for node/graph encoding, Customized Settings include Dimension, Layer, and GNN-Type"""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layer=2, gnn_type="GAT"):
        super().__init__()

        if gnn_type == "GCN":
            conv = GCNConv
        elif gnn_type == "GAT":
            conv = GATConv
        elif gnn_type == "TransformerConv":
            conv = TransformerConv
        elif gnn_type == "SAGE":
            conv = SAGEConv
        else:
            raise KeyError("GNN_TYPE can only be GAT, GCN, SAGE, and TransformerConv")

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act = torch.nn.LeakyReLU()
        
        # Sum Pooling 
        self.pool = global_add_pool

        if n_layer < 2:
            raise ValueError('GNN layer_num should >=2 but you set {}'.format(n_layer))
        elif n_layer == 2:
            self.conv_layers = torch.nn.ModuleList([conv(input_dim, hidden_dim), conv(hidden_dim, output_dim)])
        else:
            layers = [conv(input_dim, hidden_dim)]
            for _ in range(n_layer - 2):
                layers.append(conv(hidden_dim, hidden_dim))
            layers.append(conv(hidden_dim, output_dim))
            self.conv_layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index, batch=None):
        for graph_conv in self.conv_layers[0:-1]:
            x = graph_conv(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, training=self.training)

        node_emb = self.conv_layers[-1](x, edge_index)

        if batch is None:
            # input is a whole graph, return all nodes' embeddings
            return node_emb

        graph_emb = self.pool(node_emb, batch)
        return graph_emb
    



class CodeProjection(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,code_tokens):
        super(CodeProjection, self).__init__()
        self.code_tokens = code_tokens
        self.out_features = out_features
        self.projection1 = nn.Linear(in_features, hidden_features)  # 第一个投影层
        self.projection2 = nn.Linear(hidden_features, code_tokens*out_features)  # 第二个投影层
        self.dropout = nn.Dropout(p=0.1)  # Dropout层，p是丢弃概率

    def forward(self, x):
        x = F.relu(self.projection1(x))  # ReLU激活函数
        x = self.dropout(x)  # 应用Dropout
        x = self.projection2(x)
        
        batch_size = x.size(0)
        return x.view(batch_size, self.code_tokens, self.out_features)