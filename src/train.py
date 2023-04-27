import os
import torch
import time
import numpy as np
from torch.data.utils import Dataset, Sampler, DataLoader
from .config import Config
from .dataset import WillyDataset, CustomSampler
from .models import  Final
from .trainer import Trainer
from .utils import *
import wandb

# Get the tokenized file 
def run_():
    path= './'
    all_files= os.listdir()
    all_files= [os.path.join(path, x) for x in all_files if x.endswith('.pt')]

    model= Final(Config).to(Config.device)
    optim= torch.optim.AdamW(model.parameters(), lr= Config.lr, weight_decay= Config.wd)

    # Trainer Baseline model invoked
    trainer= Trainer(model, optim, loss_fn, wandb= Config.wandb)
    val_loss_min= np.Inf
    if Config.wandb:
        wandb.watch(model)
    start_time= time.time()
    fl= all_files[0]
    tokenized_file= torch.load(fl, map_location= Config.device)
    for epoch_ in range(Config.epochs):
        print(f"{'='*40} EPOCH: {epoch_+1}/{Config.epochs} {'='*40}")
        dataset= WillyDataset(tokenized_file, Config)
        # Loading the data (NOT THE BEST METHOD)
        loader= DataLoader(dataset, sampler= CustomSampler(dataset, Config.block_size, replacement=True, num_samples= Config.bs * Config.batches))
        print("Training on {os.path.basename(fl)}")
        trainer.train_one_epoch(loader)
        val_loss, op_model= trainer.valid_one_epoch(loader)
        
        # If the valid loss is decreasing, we save the model, think it as inefficeint checkpointing
        if val_loss<= val_loss_min:
            print("Valid_loss decreased ({:.4f}-->{:.4f}). Saving tthe model...".format(val_loss_min, val_loss))
            torch.save(op_model.state_dict(), f"Epoch= {epoch_+1}_{Config.model}.pt")
            val_loss_min= val_loss
    end= time.time()- start_time
    time_taken= end / (60*60)
    print(f"Time taken= {time_taken:.1f} HOURS TO TRAIN THE MODEL")

if __name__=="__main__":
    wandb.login()
    wandb.init(project='gpt2', group= 'llm', job_type= 'train', config= Config)

    a= run_()

    




