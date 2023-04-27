# Creating custom dataset to be loaded in the model
# Tokenizing is done within the data loading pipeline for a fixed window size
# This is done because GPT2 can take 2048 token at a time and tokenising all at once will be a waste of resource

from torch.utils.data import Dataset, Sampler
from .config import Config
from transformers import GPT2TokenizerFast

class WillyDataset(Dataset):
    def __init__(self, text_file, config, tokenizer=None):
        self.text_file= text_file
        self.config= config
        self.tokenizer= tokenizer if tokenizer else GPT2TokenizerFast.from_pretrained('gpt2')
    
    def __getitem__(self, idx):
        curr_window= self.text[idx: idx+self.config.block_size+1]
        tokenized_text= self.tokenizer.encode(" ".join(curr_window), return_tensors='pt')[0]
        input_text= tokenized_text[0:self.config.block_size]
        target_text= tokenized_text[1:self.config.block_size+1]
        return (input_text, target_text)
    def __len__(self):
        return len(self.text_file)

class CustomSampler(Sampler):
    def __init__(self, data, block_size, replacement= False, num_samples= None):
        self.data= data
        self.block_size= block_size
        self.replacement= replacement
        self._num_samples= num_samples
    @property
    def num_samples(self):
        self._num_samples is None:
            return len(self.data)
        return self._num_samples
    def __iter__(self):
        n= len(self.data)- self.block_size
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples
