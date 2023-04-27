from .config import Config
from .models import Final
import torch
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, optim, loss_fn, wandb= False):
        self.model= model
        self.loss_fn= loss_fn
        self.wandb= wandb
        self.optim= optim

    def train_one_epoch(self, train_loader):
        for input_, target in tqdm(train_loader, total= len(train_loader)):
            logits= self.model(input_)
            loss= self.loss_fn(logits, target)
            loss_item= loss.item()
            if self.wandb:
                wandb.log({'Train Loss': loss_item})
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()
    @torch.no_grad()
    def valid_one_epoch(self, valid_loader):
        for input_, target in tqdm(valid_loader, total= len(valid_loader)):
            logits= self.model(input_)
            loss= self.loss_fn(logits, target)
            loss_item= loss.item()
            if self.wandb:
                wandb.log({'Val_loss': loss_item})
        print(f"VLoss= {loss_item:.4f}")
        return loss_item, self.model
