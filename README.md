# WillyGPT
![Attention Architecture](/assets/arch1.png)

---
This repository implements a GPT2 model and pretraining from scratch on the William Shakepeare's plays that have been extracted from [here](https://www.thecompleteworksofshakespeare.com). I wrote a python script for scraping all the plays and created a `.txt` file for all of it.

The model is trained on the corpus of all his plays. Since pre training had to done, we ought to run the model for multiple epochs for optimum learning. Due to the resource constraints the model ran for 10 epochs, reaching the loss value of 11.8 %. 

The project has been inspired from Andrej Karpathy's [minGPT](https://www.github.com/karpathy/minGPT)

I would recommend to go through the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf) to know more about the transformer models and the significance of **Self Attention**.
## File Structure
---
```
WillGPT
├──README.md
├──requirements.txt
├──scrape.py
├──final.txt
├──LICENSE
└───assets
    └───arch1.png
└───src
    ├──config.py
    ├──dataset.py
    ├──models.py
    ├──trainer.py
    ├──utils.py
    └───train.py
```
## Data
---

The text corpus used can be found in `final.txt`. The file is a combined corpus of all the plays by shakespeare. It has been extracted by writing a python script  `scrape.py`.

## Training
---
If you want to train the model, you have to perform the following steps:

**1. Get the Data**
Download the `final.txt` file and put it within the directory  `./`

Rename the file and change it within the code accordingly. 


**2. Installing the Dependencies**

To run the code in this repository, few frameworks need to be installed in your machine. Make sure you have enough space and stable internet connection.

Run the below command for installing the required dependencies.

```shell
$ pip install -r requirements.txt 
```

**3. Configure Weights & Biases**

Create a project in your [weights and biases](https://wandb.ai/) account and when you run the `train.py` file you will be asked for the `user_secret` to configure W&B with your program. 

This step is done to track the progress of the model and log the losses. `W&B` is a brilliant platform for logging and tracking your model, giving you beautiful charts as output. 

**4. Training the model** 

If you have the above steps right then, running the train.py should not produce any errors. 
To run the code, open the terminal and change the directory level same as `train.py` file. 
Now run the `train.py` file.

```shell
$ python train.py
```
You should start seeing the progress bar, on few seconds at the beginning of training.

If you have any problem, feel free to open a issue. Will be happy to help.

**5. Inference**

Run the below code that will show you the output of the model.

```shell
from .utils import generate
from transformers import GPT2TokenizerFast
from .models import Final
from .config import Config
import torch

model= Final(config)
model.load_state_dict(torch.load('./EPOCH=10_gpt_small.pt')) # For you the pathname will be the file will lowest loss value

tokenizer= GPT2TokenizerFast.from_pretrained("gpt2")
prompt = tokenizer.encode("Thy not feel", return_tensors='pt').to(config['device'])
generated_text = generate(model, prompt, max_tokens=config['block_size'])
generated_text = tokenizer.decode(generated_text.tolist()[0])
print(generated_text)

```

I have given the prompt as **Thy not feel**. The output I got for the given prompt is shown below:

```diff
Thy not feel,
+Nor thyself to receive thine.

+PUCK

+Here live thou mock the moon:I'll give thee promise;For if thou shouldst, I'll live to prove;Or I'll live by and force perforce.

+Exit

+O the difference of her live!How in the murder of the story thou, PUCK

+Thou shalt be as thou look'st,Tie.
 
+PUCK

+I'll fee the knave.I'll keep a church and be by;I'll yonder dog;And I will go.

+Enter two Gentlemen

+First Captain

+Under this thick-room.

+Second Witch

+Thrice set forth and down and laugh to the old.

+First Drawer

+Thrice the stage.

+A bloody Child crowned.

+Third Witch

+From the tree,Thus did they bear;And morn, to meward.

+Second Witch

+Let him go:Wear our horses.

+Third Witch

+A drum and trumpets. Enter MACBETH

+A terrible, and some AttendantSecond in his hands

+MACBETH

+Thou shalt have the yet have the Third Apparition:Hang'd the snake,
```



