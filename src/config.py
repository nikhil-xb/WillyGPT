import torch 
class Config:
    block_size= 256
    n_embed= 768
    num_blocks= 12
    num_head= 12
    head_size= n_embed // num_head
    att_drop= 0.2
    multihead_drop_value= 0.2
    ffn_drop_value= 0.2
    vocab_size= 50257
    device= torch.device("cuda") if torch.device.cuda.is_available() else "cpu"
    epochs= 40
    lr= 5e-4
    wd= 1e-5
    bs= 128
    wandb= True
    model= 'gpt_small.pt'
    # put your own path in case you want to restart training after certain epoch 
    pretrained= None 
    batches= 32

