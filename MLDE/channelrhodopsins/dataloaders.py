import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchtext.transforms as T
from torchtext.vocab import build_vocab_from_iterator

from Protabank.utils import random_test_train_split

def constructVocab(dataset):
    def getTokens(iter):
        for ite in iter: return [ch for ch in ite]
    
    return build_vocab_from_iterator(
        getTokens(list(dataset['Sequence'])),
        specials= ['<sos>', '<eos>'],
        special_first=True
    )

class SeqToFuncDataset(Dataset):
    def __init__(self, sequences, targets, vocab, max_length):
        self.sequences = sequences
        self.targets = targets
        self.vocab = vocab
        self.max_length = max_length
        self.text_transform = self.get_transform(vocab)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]

        # Tokenize, one-hot encode, and pad the sequence
        encoded_sequence = self.tokenize_and_encode(sequence)
        padded_sequence = self.pad_sequence(encoded_sequence, self.max_length)

        return {'sequence': padded_sequence, 'target': target}
    
    # Turns out that we dont one-hot encode any padding to 0 terms.
    def get_transform(self, vocab):
        text_transform = T.Sequential(
            ## converts the sentences to indices based on given vocabulary
            T.VocabTransform(vocab=vocab),
            ## Add <sos> at the beginning of each sentence
            T.AddToken(0, begin=True),
            ## Add <eos> at the end of each sentence
            T.AddToken(1, begin=False)
        )
        return text_transform

    def tokenize_and_encode(self, sequence):
        # Use the provided text_transform to tokenize and encode the sequence
        '''
        print(sequence)
        if not isinstance(sequence, list):
            raise TypeError("Input sequence must be a list of tokens.")
        '''
        tokenized_sequence = self.text_transform(list(sequence))
        # Convert the tokens to one-hot encoding
        encoded_sequence = F.one_hot(torch.tensor(tokenized_sequence), num_classes=len(self.vocab))
        return encoded_sequence.float()

    def pad_sequence(self, sequence, max_length):
        # Implement padding logic to make all sequences the same length
        current_length = sequence.size(0)
        if current_length < max_length:
            padding_length = max_length - current_length
            padding = torch.zeros(padding_length, sequence.size(1))
            padded_sequence = torch.cat([sequence, padding])
        else:
            padded_sequence = sequence[:max_length]

        return padded_sequence
    
def checkTransform(dataset):
    import random
    from Protabank.utils import aa_map
    vocab = constructVocab(dataset)
    Q = SeqToFuncDataset(dataset['Sequence'][:10], dataset['Data'][:10], vocab = vocab, max_length=300)
    seq = [random.choice(list(aa_map.keys())) for _ in range(10)]
    print("Original: {}".format(seq))
    encoded = Q.get_transform(vocab)(seq)
    print("Encoded: {}".format(encoded))
    # oneHotted = Q.tokenize_and_encode("".join(seq))
    # print("One-Hotted: {}".format(oneHotted))
    print("Re-Transformed: {}".format(" ".join(vocab.get_itos()[index] for index in encoded)))
    
# dataset is a df with columns Sequence, Data
def get_dataloaders(dataset, batch_size = 16, split_percent = 0.8, max_length = 1000):
    assert len(dataset) > 10 # prevents any funky business
    
    train_data, test_data, train_target, test_target = random_test_train_split(
        dataset, target_column='Data', test_size=0.2, random_seed=42
    )
    
    vocab = constructVocab(dataset)
    train_dataset = SeqToFuncDataset(
        list(train_data), list(train_target), vocab = vocab, max_length=max_length)
    test_dataset = SeqToFuncDataset(
        list(test_data), list(test_target), vocab = vocab, max_length=max_length)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_dataloader, test_dataloader

# checkTransform(dataset)