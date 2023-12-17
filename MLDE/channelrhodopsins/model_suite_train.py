# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 21:12:38 2023

@author: Bill Sun
"""
import pandas as pd
import numpy as np

from MLDE.channelrhodopsins.dataloaders import get_dataloaders, constructVocab
from MLDE.models.MLP import *
from MLDE.models.LSTM import *
from MLDE.channelrhodopsins.train import train
from Protabank.utils import plotLosses

from MLDE.channelrhodopsins.data_preprocessing import clean

PATH = './Data/ChR/pnas.1700269114.sd01.csv'

dataset = pd.read_csv(PATH)
dataset = dataset[['sequence','mKate_mean','GFP_mean', 'intensity_ratio_mean']]
dataset = clean(dataset, 'GFP_mean')
# %%
BATCH_SIZE = 8
vocab = constructVocab(dataset) # should be 4 Nucleotides's + <sos> + <eos>
# print(vocab.get_itos())
# df['Sequence'].apply(lambda x: len(str(x))).describe()
max_length = 1100 # somewhat arbitrarily chosen
train_dataloader, test_dataloader = get_dataloaders(dataset, batch_size=BATCH_SIZE, split_percent = 0.8, max_length = max_length)
# %%
print(len(train_dataloader))
print(len(test_dataloader))
# %%
input_size = max_length * len(vocab)

models = {"LR":LR(input_size = input_size, output_size = 1),
          "MLP_sm":MLP1(input_size = input_size, output_size = 1),
          "MLP_md":MLP2(input_size = input_size, output_size = 1),
          "MLP_lg":MLP3(input_size = input_size, output_size = 1)}

lstm_models = {
    "BLSTM1":BLSTM1(input_size = input_size, output_size = 1),
    "LSTM1":LSTM1(input_size = input_size, output_size = 1),
          # "LSTM_BN1":LSTM_BN1(input_size = input_size, output_size = 1)
          }
# %%
import matplotlib.pyplot as plt

# Your dictionary mapping experiment to train and val losses
loss_data = {
    # Add more experiments as needed
}

for model_name, model in models.items():
    print("<" + "-"*25 + ">")
    train_losses, test_losses = train(model, train_dataloader, test_dataloader, verbose = True, num_epochs = 50)
    # plotLosses(model_name, train_losses, test_losses)
    loss_data[model_name] = {'train': train_losses, 'val': test_losses}
    print("{} Loss: {}".format(model_name, test_losses[-1]))
# %%
# Create subplots with a single row
num_experiments = len(loss_data)
fig, axes = plt.subplots(ncols=num_experiments, figsize=(4 * num_experiments, 4))

# Iterate through experiments and plot on subplots
for i, (experiment, losses) in enumerate(loss_data.items()):
    # Plot training loss
    axes[i].plot(losses['train'], label='Train Loss', marker='o')
    
    # Plot validation loss
    axes[i].plot(losses['val'], label='Validation Loss', marker='o')
    
    # Set subplot title and labels
    axes[i].set_title(experiment)
    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Loss')
    
    # Display legend
    axes[i].legend()

# Adjust layout to prevent overlapping
plt.tight_layout()
# plt.suptitle("MLPs Performance in Predicting Expression")
# Show the plot
plt.show()
