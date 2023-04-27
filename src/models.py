'''
This file contains the implementation of the transformer architecture.
The model implementation has been greatly influenced by Andrej Karapathy's MiniGPT.
To know more about transformers look at, "Attention Is All You Need" https://arxiv.org/abs/1706.03762
To know more about GPT-2 model look at, https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

The model has been structured in the following manner:
    1. Single Self Attention Model
    2. Multihead Self Attention in the FeedForward block
    3. The GPT-2 Block 
'''
from .config import Config
import torch
from torch.nn as nn
import math

class SelfAttentionHead(nn.Module):
    def __init__(self, config):
        super(SelfAttentionHead, self).__init__()
        self.config= config
        self.n_head= config.n_head
        self.n_embed= config.n_embed
        self.c_attn= nn.Linear(config.n_embed, 3*config.n_embed)
        self.c_proj= nn.Linear(config.n_embed, config.n_embed)
        self.att_drop= nn.Dropout(config.att_drop)
        self.resdrop= nn.Dropout(config.multihead_drop_value)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)).view(1,1,config.block_size, config.block_size))

    def forward(self,x):
        B, T, C= x.size()
        q,k,v= self.c_attn(x).split(self.n_embed, dim=2)
        k= k.view(B, T, self.n_head, C// self.n_head).transpose(1,2)
        q= q.view(B, T, self.n_head, C// self.n_head).transpose(1,2)
        v= v.view(B, T, self.n_head, C// self.n_head).transpose(1,2)
        wei= (q @ k.transpose(-2,-1))* (1/math.sqrt(k.size(-1)))
        wei= wei.masked_fill(self.tril[:,:,:T,:T]==0, float('-inf'))
        wei= F.softmax(wei, dim=-1)
        wei= self.att_drop(wei)
        out= wei @ v
        out= out.transpose(1,2).contiguous().view(B,T,C)
        out= self.resdrop(self.c_proj(out))
        return out
 
class Block(nn.Module):
    def __init__(self, config):
         super(Block,self).__init__()
         self.ln1= nn.LayerNorm(config.n_embed)
         self.attn= SelfAttentionHead(config)
         self.ln2= nn.LayerNorm(config.n_embed)
         self.mlp= nn.ModuleDict(dict(
             c_fc= nn.Linear(config.n_embed, config.n_embed*4),
             c_proj= nn.Linear(config.n_embed*4, config.n_embed),
             act= nn.GELU(),
             dropout= nn.Dropout(config.ffn_drop_value)
             ))
         m= self.mlp
         self.mlp= lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))
    def forward(self,x):
        x= x+ self.attn(self.ln1(x))
        x= x+ self.mlp(self.ln2(x))
        return x

class Final(nn.Module):
    def __init__(self, config):
        super(Final, self).__init__()
        self.config= config
        self.token_embedding_table= nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table= nn.Embedding(config.block_size,config.n_embed)
        self.block= nn.Sequential(*[Block(config) for _ in range(config.num_blocks)])
        self.ln_f= nn.LayerNorm(config.n_embed)
        self.lm_head= nn.Linear(config.n_embed,config.vocab_size,bias= False)
    def forward(self,x,targets= None):
        B,T= x.shape
        tok_emb= self.token_embedding_table(x) # (B,T,C)
        pos_emb= self.position_embedding_table(torch.arange(T, device=self.config.device,dtype= torch.long)).unsqueeze(0)
        x= tok_emb+ pos_emb
        x= self.block(x)
        x= self.ln_f(x)
        logits= self.lm_head(x)
        return logits

