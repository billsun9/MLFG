import matplotlib.pyplot as plt
import numpy as np

# weird matplotlib error
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

s = '''
A	Alanine	
C	Cysteine	
D	Aspartic acid	
E	Glutamic acid	
F	Phenylalanine	
G	Glycine	
H	Histidine	
I	Isoleucine	
K	Lysine	
L	Leucine	
M	Methionine	
N	Asparagine	
P	Proline	
Q	Glutamine	
R	Arginine	
S	Serine	
T	Threonine	
V	Valine	
W	Tryptophan	
Y	Tyrosine	
'''

def constructAminoAcidMap(s):
    
    s = s.split("\n")
    s = [x.split("\t")[:-1] for x in s]
    s = {x[0]: x[1] for x in s if x}
    return s

aa_map = constructAminoAcidMap(s)

def plotLosses(name, train_loss, val_loss):
    plt.plot(list(range(0, len(train_loss))), train_loss, label='train_loss')
    plt.plot(list(range(0, len(train_loss))), val_loss, label='val_loss')
    
    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('{}: Epoch vs MSE'.format(name))
    
    # Adding a legend to distinguish between the two lists
    plt.legend()
    
    # Display the plot
    plt.show()
