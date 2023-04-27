import torch
import numpy as np
import torch.nn as nn
from .config import Config

def loss_fn(logits, targets):
    logits= logits.view(-1, logits.size(-1))
    targets= targets.view(-1)
    loss= nn.functional.cross_entropy(logits, targets)
    return loss

def generate(model, prompt, max_tokens, temp= 0.7):
    for _ in range(max_tokens):
        prompt= prompt[:,:Config.block_size]
        logits= model(prompt)
        logits= logits[:,-1, :]
        logits= logits/temp
        probs= nn.functional.softmax(logits, dim= -1)
        next_prompt= torch.multinomial(probs, num_samples= 1)
        prompt= torch.cat((prompt, next_prompt), dim= 1)
    return prompt
