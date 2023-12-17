class NucleotideToAA:
    def __init__(self):

        self.codon_map = {
            'TCA': 'S',    # Serina
            'TCC': 'S',    # Serina
            'TCG': 'S',    # Serina
            'TCT': 'S',    # Serina
            'TTC': 'F',    # Fenilalanina
            'TTT': 'F',    # Fenilalanina
            'TTA': 'L',    # Leucina
            'TTG': 'L',    # Leucina
            'TAC': 'Y',    # Tirosina
            'TAT': 'Y',    # Tirosina
            'TAA': '*',    # Stop
            'TAG': '*',    # Stop
            'TGC': 'C',    # Cisteina
            'TGT': 'C',    # Cisteina
            'TGA': '*',    # Stop
            'TGG': 'W',    # Triptofano
            'CTA': 'L',    # Leucina
            'CTC': 'L',    # Leucina
            'CTG': 'L',    # Leucina
            'CTT': 'L',    # Leucina
            'CCA': 'P',    # Prolina
            'CCC': 'P',    # Prolina
            'CCG': 'P',    # Prolina
            'CCT': 'P',    # Prolina
            'CAC': 'H',    # Histidina
            'CAT': 'H',    # Histidina
            'CAA': 'Q',    # Glutamina
            'CAG': 'Q',    # Glutamina
            'CGA': 'R',    # Arginina
            'CGC': 'R',    # Arginina
            'CGG': 'R',    # Arginina
            'CGT': 'R',    # Arginina
            'ATA': 'I',    # Isoleucina
            'ATC': 'I',    # Isoleucina
            'ATT': 'I',    # Isoleucina
            'ATG': 'M',    # Methionina
            'ACA': 'T',    # Treonina
            'ACC': 'T',    # Treonina
            'ACG': 'T',    # Treonina
            'ACT': 'T',    # Treonina
            'AAC': 'N',    # Asparagina
            'AAT': 'N',    # Asparagina
            'AAA': 'K',    # Lisina
            'AAG': 'K',    # Lisina
            'AGC': 'S',    # Serina
            'AGT': 'S',    # Serina
            'AGA': 'R',    # Arginina
            'AGG': 'R',    # Arginina
            'GTA': 'V',    # Valina
            'GTC': 'V',    # Valina
            'GTG': 'V',    # Valina
            'GTT': 'V',    # Valina
            'GCA': 'A',    # Alanina
            'GCC': 'A',    # Alanina
            'GCG': 'A',    # Alanina
            'GCT': 'A',    # Alanina
            'GAC': 'D',    # Acido Aspartico
            'GAT': 'D',    # Acido Aspartico
            'GAA': 'E',    # Acido Glutamico
            'GAG': 'E',    # Acido Glutamico
            'GGA': 'G',    # Glicina
            'GGC': 'G',    # Glicina
            'GGG': 'G',    # Glicina
            'GGT': 'G'     # Glicina
        }

    def translate_dna_to_amino_acids(self, dna_sequence):
        if len(dna_sequence) % 3 != 0:
            raise ValueError("Length of the DNA sequence must be a multiple of 3.")
    
        amino_acids = []
    
        for i in range(0, len(dna_sequence), 3):
            codon = dna_sequence[i:i+3]
            if codon not in self.codon_map: raise ValueError("{} is invalid codon".format(codon))
            amino_acid = self.codon_map[codon]
            if amino_acid == '*': break
            amino_acids.append(amino_acid)
    
        return ''.join(amino_acids)
    
    # Example usage:
    def ex(self):
        dna_sequence = "TTTTAAGATGAT"
        amino_acids_result = self.translate_dna_to_amino_acids(dna_sequence)
        print("DNA sequence:", dna_sequence)
        print("Amino acids:", amino_acids_result)

        
# # %%
# import sys
# import os
# from collections import defaultdict
# import matplotlib.pyplot as plt

# CGR_X_MAX = 1
# CGR_Y_MAX = 1
# CGR_X_MIN = 0
# CGR_Y_MIN = 0
# CGR_A = (CGR_X_MIN, CGR_Y_MIN)
# CGR_T = (CGR_X_MAX, CGR_Y_MIN)
# CGR_G = (CGR_X_MAX, CGR_Y_MAX)
# CGR_C = (CGR_X_MIN, CGR_Y_MAX)
# CGR_CENTER = ((CGR_X_MAX - CGR_Y_MIN) / 2, (CGR_Y_MAX - CGR_Y_MIN) / 2)

# def empty_dict():
# 	"""
# 	None type return vessel for defaultdict
# 	:return:
# 	"""
# 	return None


# CGR_DICT = defaultdict(
# 	empty_dict,
# 	[
# 		('A', CGR_A),  # Adenine
# 		('T', CGR_T),  # Thymine
# 		('G', CGR_G),  # Guanine
# 		('C', CGR_C),  # Cytosine
# 		('U', CGR_T),  # Uracil demethylated form of thymine
# 		('a', CGR_A),  # Adenine
# 		('t', CGR_T),  # Thymine
# 		('g', CGR_G),  # Guanine
# 		('c', CGR_C),  # Cytosine
# 		('u', CGR_T)  # Uracil/Thymine
# 		]
# )

# def mk_cgr(seq):
# 	"""Generate cgr

# 	:param seq: list of nucleotide
# 	:return cgr: [['nt', (x, y)]] List[List[Tuple(float, float)]]
# 	"""
# 	cgr = []
# 	cgr_marker = CGR_CENTER[:
# 		]    # The center of square which serves as first marker
# 	for s in seq:
# 		cgr_corner = CGR_DICT[s]
# 		if cgr_corner:
# 			cgr_marker = (
# 				(cgr_corner[0] + cgr_marker[0]) / 2,
# 				(cgr_corner[1] + cgr_marker[1]) / 2
# 			)
# 			cgr.append([s, cgr_marker])
# 		else:
# 			sys.stderr.write("Bad Nucleotide: " + s + " \n")

# 	return cgr


# def mk_plot(cgr, name, figid):
# 	"""Plotting the cgr
# 		:param cgr: [(A, (0.1, 0.1))]
# 		:param name: str
# 		:param figid: int
# 		:return dict: {'fignum': figid, 'title': name, 'fname': helper.slugify(name)}
# 	"""
# 	x_axis = [i[1][0] for i in cgr]
# 	y_axis = [i[1][1] for i in cgr]
# 	plt.figure(figid)
# 	plt.title("Chaos Game Representation\n" + name, wrap=True)
# 	# diagonal and vertical cross
# 	# plt.plot([x1, x2], [y1, y2])
# 	# plt.plot([0.5,0.5], [0,1], 'k-')
# 	plt.plot([CGR_CENTER[0], CGR_CENTER[0]], [0, CGR_Y_MAX], 'k-')

# 	# plt.plot([0,1], [0.5,0.5], 'k-')
# 	plt.plot([CGR_Y_MIN, CGR_X_MAX], [CGR_CENTER[1], CGR_CENTER[1]], 'k-')
# 	plt.scatter(x_axis, y_axis, alpha=0.5, marker='.')

# 	return {'fignum': figid, 'title': name, 'fname': name}


# def write_figure(fig, output_dir, dpi=300):
# 	"""Write plot to png
# 	:param fig:  {'fignum':figid, 'title':name, 'fname':helper.slugify(name)}
# 	:param dpi: int dpi of output
# 	:param output_dir: str

# 	Usage:
# 		figures = [mk_plot(cgr) for cgr in all_cgr]
# 		for fig in figures:
# 			write_figure(fig, "/var/tmp/")
# 		The figid in the mk_plot's return dict must be present in plt.get_fignums()
# 	"""
# 	all_figid = plt.get_fignums()
# 	if fig['fignum'] not in all_figid:
# 		raise ValueError("Figure %i not present in figlist" % fig['fignum'])
# 	plt.figure(fig['fignum'])
# 	target_name = os.path.join(
# 		output_dir,
# 		fig['fname'] + ".png"
# 	)
# 	plt.savefig(target_name, dpi=dpi)
    