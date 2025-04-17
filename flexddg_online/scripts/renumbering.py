import argparse
import abnumber
from Bio import PDB
from Bio.PDB import Model, Chain, Residue, Selection
from Bio.SeqUtils import seq1
from typing import List, Tuple


def biopython_chain_to_sequence(chain: Chain.Chain):
    residue_list = Selection.unfold_entities(chain, 'R')
    seq = ''.join([seq1(r.resname) for r in residue_list])
    return seq, residue_list


def assign_number_to_sequence(seq, scheme='chothia'):
    abchain = abnumber.Chain(seq, scheme=scheme)
    offset = seq.index(abchain.seq)
    if not (offset >= 0):
        raise ValueError(
            'The identified Fv sequence is not a subsequence of the original sequence.'
        )

    numbers = [None for _ in range(len(seq))]
    for i, (pos, aa) in enumerate(abchain):
        resseq = pos.number
        icode = pos.letter if pos.letter else ' '
        numbers[i+offset] = (resseq, icode)
    return numbers, abchain


def renumber_biopython_chain(chain_id, residue_list: List[Residue.Residue], numbers: List[Tuple[int, str]]):
    chain = Chain.Chain(chain_id)
    for residue, number in zip(residue_list, numbers):
        if number is None:
            continue
        residue = residue.copy()
        new_id = (residue.id[0], number[0], number[1])
        residue.id = new_id
        chain.add(residue)
    return chain


def renumber(in_pdb, out_pdb, scheme='chothia', return_other_chains=False):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(None, in_pdb)
    model = structure[0]
    model_new = Model.Model(0)

    heavy_chains, light_chains, other_chains = [], [], []

    for chain in model:
        try:
            seq, reslist = biopython_chain_to_sequence(chain)
            numbers, abchain = assign_number_to_sequence(seq, scheme=scheme)
            chain_new = renumber_biopython_chain(chain.id, reslist, numbers)
            print(f'[INFO] Renumbered chain {chain_new.id} ({abchain.chain_type})')
            if abchain.chain_type == 'H':
                heavy_chains.append(chain_new.id)
            elif abchain.chain_type in ('K', 'L'):
                light_chains.append(chain_new.id)
        except abnumber.ChainParseError as e:
            print(f'[INFO] Chain {chain.id} does not contain valid Fv: {str(e)}')
            chain_new = chain.copy()
            other_chains.append(chain_new.id)
        model_new.add(chain_new)

    pdb_io = PDB.PDBIO()
    pdb_io.set_structure(model_new)
    pdb_io.save(out_pdb)
    if return_other_chains:
        return heavy_chains, light_chains, other_chains
    else:
        return heavy_chains, light_chains

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pattern", type=str, default="*.pdb", help="Pattern to find input PDB files (e.g. *.pdb)")
    parser.add_argument("--output_suffix", type=str, default="renumbered")
    parser.add_argument("--force", action="store_true", help="Force renumbering even if the chain is not Fv")
    parser.add_argument("--scheme", type=str, default="imgt", choices=["chothia", "kabat", "imgt"], help="Numbering scheme to use")
    args = parser.parse_args()
    # find all pdb files
    import os, glob
    pdb_files = glob.glob(args.input_pattern)

    print(f'Found {len(pdb_files)} PDB files to renumber')

    for pdb_file in pdb_files:
        output_path = os.path.splitext(pdb_file)[0] + '_' + args.output_suffix + '.pdb'
        if os.path.exists(output_path) and not args.force:
            print(f'Skipping {pdb_file} because renumbered file already exists')
            continue
        try:
            renumber(pdb_file, output_path, scheme=args.scheme)
            print(f'Successfully renumbered {pdb_file}')
        except Exception as e:
            print(f'[ERROR] Failed to renumber {pdb_file}: {str(e)}')
