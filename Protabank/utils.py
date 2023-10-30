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