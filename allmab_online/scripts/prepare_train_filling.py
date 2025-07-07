#!/usr/bin/python3

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing
from utils import load_config


def mutations_to_seq(mutations, wt_seq, offset=0, indel_indices=None):
    if indel_indices is None:
        indel2indices = {i+offset:i for i in range(len(wt_seq))}
    else:
        indel2indices = {v:i for i,v in enumerate(indel_indices)}
    mutseq = list(wt_seq)

    for mutation in mutations:
        # wt, pos, mut = mutation[0], int(mutation[2:-1]) - offset, mutation[-1]
        wt, pos, mut = mutation[0], int(indel2indices[int(mutation[1:-1])]), mutation[-1]
        assert wt == mutseq[pos], f"{wt}!={mutseq[pos]}, {mutation}, {mutseq}"
        mutseq[pos] = mut
    return ''.join(mutseq)

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ddG output')
    parser.add_argument('--config', type=str)
    parser.add_argument('--mutations_file', type=str)
    parser.add_argument('--output_dir', type=str, help='output directory')
    parser.add_argument('--previous_data', type=str, help='previous data to use')
    parser.add_argument('--cycle', type=int, help='cycle number', default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading config...")
    config = load_config(args.config)

    # Create output directories
    print("Creating output directories...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    wildtype_sequence = config["wildtype_sequence"]
    wildtype_sequence = {k: v for k, v in wildtype_sequence.items() if k in config["mutable_chain"]}
    # Prepare training data if needed
    training_file = os.path.join(args.output_dir, "training_data_subset.csv")
    if not os.path.exists(training_file):
        print("Preparing training data...")
        mutations_df = pd.read_csv(args.mutations_file)
        
        # Use DMS_score from mutations file as ddG scores
        mutations_df["ddG"] = mutations_df["DMS_score"]
        chains = list(config["mutable_chain"])
        mutable_chain = config["mutable_chain"]
        assert len(chains) == 1, "Only one chain is supported"
        # Process mutations
        print("Processing mutations...")
        mutant_dict_list = []
        mutated_sequence_list = []
        mutant_str_dict_list = []
        for idx, row in tqdm(mutations_df.iterrows(), total=len(mutations_df), desc="Processing mutations"):
            mutant_dict = {chain: [] for chain in chains}
            mutations = row["mutations"]
            if pd.isna(mutations):
                mutations = []
            else:
                mutations = mutations.split(",")
            for mutant in mutations:
                mutant_dict[mutable_chain].append(f"{mutant[0]}{mutant[1:-1]}{mutant[-1]}")
            mutant_str_dict = {}
            mutated_sequence = {}
            for chain in chains:
                seq = wildtype_sequence[chain]
                mutated_sequence[chain] = mutations_to_seq(mutant_dict[chain], seq, offset=1)
                mutant_str_dict[chain] = ":".join(mutant_dict[chain])
            mutant_dict_list.append(mutant_dict)
            mutated_sequence_list.append(mutated_sequence)
            mutant_str_dict_list.append(mutant_str_dict)

        # Add metadata
        print("Adding metadata...")
        mutations_df = mutations_df.assign(
            POI=config["POI"],
            DMS_score=-mutations_df["ddG"],
            mutant=mutant_str_dict_list,
            wildtype_sequence=[wildtype_sequence.copy() for _ in range(len(mutations_df))],
            mutated_sequence=mutated_sequence_list,
            chain_id=config["mutable_chain"],
            pdb_file=config["POI"],
            cycle=args.cycle
        )
        mutations_df = mutations_df[["POI", "DMS_score", "mutant", "wildtype_sequence", "mutated_sequence", "chain_id", "pdb_file", "cycle", "mutseq", "mutations"]]
        if config.get("model_type", "sequence") in ["ablang2", "ablang_gp", "blosum_gp"]:
            mutations_df["wildtype_heavy"] = mutations_df["wildtype_sequence"].apply(lambda x: x[config["mutable_chain"]])
            mutations_df["wildtype_light"] = config["light_wt_sequence"]
            mutations_df["heavy"] = mutations_df["mutated_sequence"].apply(lambda x: x[config["mutable_chain"]])
            mutations_df["light"] = config["light_wt_sequence"]

        # Save training data
        print(f"Saving training data to {training_file}")
        mutations_df.to_csv(training_file, index=False)

    # Combine with previous data if available
    print("Combining training data...")
    all_data = pd.read_csv(training_file)
    if args.previous_data:
        print(f"Loading and merging previous data from {args.previous_data}")
        previous_data = pd.read_csv(args.previous_data)
        all_data = pd.concat([previous_data, all_data])

    output_file = os.path.join(args.output_dir, f"training_data.csv")
    print(f"Saving combined training data to {output_file}")
    all_data.to_csv(output_file, index=False)
    print("Done!")

if __name__ == '__main__':
    main()
