import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import wandb
wandb.login()


params = dict(
    WANDB_PROJECT="pytorch-blueprint",
    ENTITY="ba-zhaw",  # team name or NONE
    RAW_DATA_AT='bdd_simple_1k',
    PROCESSED_DATA_AT='bdd_simple_1k_split'
)

'''
Following line is just for upload. If Model config is present, use like this:
Check Jupyter Notebook
config = dict(
    epochs=5,
    classes=10,
    kernels=[16, 32],
    batch_size=128,
    learning_rate=0.005,
    dataset="MNIST",
    architecture="CNN")
run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="upload", config=hyperparameters, dir='./testDir'):
'''
#run = wandb.init(project=params.WANDB_PROJECT, entity=params.ENTITY, job_type="upload")
data = pd.read_csv("./data/VDJdb/VDJdb_data.tsv", sep="\t", nrows=100)
print(data.shape)
#run.finish()
