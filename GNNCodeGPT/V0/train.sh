#!/bin/bash
export model_path="/mnt/disk2/project2/Soft_Prompt_adapter/GPT2"
export pretra_gnn="2"

export tuned_proj="2"
export train_path="/mnt/disk2/project2/Soft_Prompt_adapter/RAG_Data/conala/conala_train.json"
export eval_path="/mnt/disk2/project2/Soft_Prompt_adapter/RAG_Data/conala/conala_eval.json"
export t_graph_path="/mnt/disk2/project2/Soft_Prompt_adapter/RAG_Data/conala/Train_graphs.pt" 
export e_graph_path="/mnt/disk2/project2/Soft_Prompt_adapter/RAG_Data/conala/Eval_graphs.pt"
export output_model="/mnt/disk2/project2/Soft_Prompt_adapter/V0/model/test.pth"




python main.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --node_features 1024\
    --gnn_hidden_dim 2056\
    --gnn_out_dim 1024\
    --n_layers 2\
    --gnn_type GAT\
    --pro_in_features 1024\
    --pro_hidden_features 512\
    --code_tokens 128\
    --train_path ${train_path}\
    --eval_path ${eval_path}\
    --t_graph_path ${t_graph_path} \
    --e_graph_path ${e_graph_path}
    --block_size 512\
    --mode train\
    --batch_size 2\
    --lr 0.0001\
    --epoch 3\
    --output_dir ${output_model}