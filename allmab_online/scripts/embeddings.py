import ablang2
from Bio.Align import substitution_matrices
import numpy as np

def blosum_embedding(sequence):
    # Load BLOSUM62 matrix
    blosum62 = substitution_matrices.load("BLOSUM62")
    aa_order = blosum62.alphabet[:20]

    # Expand to 21x21 (add zeros for separator)
    matrix = substitution_matrices.load("BLOSUM62")
    matrix=np.array(matrix)
    matrix = matrix[:20][:,:20]

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Flip spectrum transformation: U * sqrt(|D|)
    transformed = eigenvectors @ np.diag(np.sqrt(np.abs(eigenvalues)))

    # Map amino acids and separator
    char_to_idx = {aa: idx for idx, aa in enumerate(aa_order)}

    # Encode sequence
    encoded = []
    for char in sequence:
        idx = char_to_idx.get(char, 0)  # Unknown chars map to AA
        encoded.append(transformed[idx])

    return np.concatenate(encoded).flatten()

def ablang_embedding(paired_sequences):
    ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, device='cuda')
    embeddings = ablang(paired_sequences, mode='seqcoding')
    return embeddings
