import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator

from Protabank.data_processing import df as dataset

BATCH_SIZE = 8

class SeqToFuncDataset(Dataset):
    def __init__(self, nucleotides, fitness, max_seq_len = 1000):
        self.nucleotides = nucleotides
        self.fitness = fitness
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.fitness)
    
    def __getitem__(self, idx):
        X, Y = self.nucleotides.iloc[idx], self.fitness.iloc[idx]
        if self.transform: X = self.transform(X)
        return X, Y

# dataset is a df with columns Sequence, Data
def get_dataloaders(dataset, split_percent = 0.8):
    assert len(dataset) > 10 # prevents any funky business
    IDX = int(split_percent * len(dataset))
    
    train_dataset = SeqToFuncDataset(
        dataset['Sequence'][:IDX], dataset['Data'][:IDX], transform=applyNucleotideTransform)
    test_dataset = SeqToFuncDataset(
        dataset['Sequence'][IDX:], dataset['Data'][IDX:], transform=applyNucleotideTransform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    return train_dataloader, test_dataloader

def constructVocab():
    def getTokens(iter):
        for ite in iter: return [ch for ch in ite]
    
    return build_vocab_from_iterator(
        getTokens(list(dataset['Sequence'])),
        specials= ['<pad>', '<sos>', '<eos>'], # THIS ORDERING MATTERS
        special_first=True
    )

def getTransform(vocab):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        ## converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        ## Add <sos> at beginning of each sentence
        T.AddToken(1, begin=True),
        ## Add <eos> at beginning of each sentence
        T.AddToken(2, begin=False)
    )
    return text_tranform

source_vocab = constructVocab()

def checkTransform():
    import random
    from Protabank.utils import aa_map
    
    seq = [random.choice(list(aa_map.keys())) for _ in range(6)]
    print("Original: {}".format(seq))
    seq = getTransform(source_vocab)(seq)
    print("Encoded: {}".format(seq))
    index_to_string = source_vocab.get_itos()
    print("Re-Transformed: {}".format(" ".join(index_to_string[index] for index in seq)))
    
# checkTransform()

def applyNucleotideTransform(sequence):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """
    x = getTransform(source_vocab)([ch for ch in sequence])
    print(x)
    return x
# %%
train_dataloader, test_dataloader = get_dataloaders(dataset)
#%%

for batch in train_dataloader:
    print(batch)
    break