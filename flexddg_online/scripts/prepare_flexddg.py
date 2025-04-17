import os
import argparse
import pandas as pd
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Create input files for Flex ddG')
    parser.add_argument('--mutations_file', type=str, help='mutations file')
    parser.add_argument('--pdbfile', type=str, default="")
    parser.add_argument('--name_col', type=str, default="name", help='name column in mutations file')
    parser.add_argument('--mutations_col', type=str, default="mutations")
    parser.add_argument('--chains_to_move', type=str, default="", help="Comma separated list of chains to move")
    parser.add_argument('--output_dir', type=str, help='output directory')
    return parser.parse_args()

def main():
    args = parse_args()
    print("Loading mutations file...")
    mutations_df = pd.read_csv(args.mutations_file, sep=",")

    print("Preparing data...")
    if args.pdbfile:
        mutations_df["pdbfile"] = args.pdbfile
    elif "pdbfile" not in mutations_df.columns:
        raise ValueError("No pdbfile column in mutations file")

    if args.name_col not in mutations_df.columns:
        mutations_df[args.name_col] = mutations_df.index

    if args.chains_to_move:
        mutations_df["chains_to_move"] = args.chains_to_move
    elif "chains_to_move" not in mutations_df.columns:
        raise ValueError("No chains_to_move column in mutations file")
    mutations_df["chains_to_move"] = mutations_df["chains_to_move"].apply(lambda x: x.split(","))

    print("Creating output files...")
    for _, row in tqdm(mutations_df.iterrows(), total=len(mutations_df), desc="Processing mutations"):
        output_dir = f"{args.output_dir}/{row[args.name_col]}"
        os.makedirs(output_dir, exist_ok=True)

        # Create resfile
        resfile_text = "NATAA\nstart\n"
        chain = row["chains_to_move"][0]
        assert len(chain) == 1, f"Only one chain is supported for now, got {chain}"

        if pd.isna(row[args.mutations_col]):
            mutations = []
        else:
            mutations = row[args.mutations_col].split(",")

        for m in mutations:
            resnum = m[1:-1]
            mut = m[-1]
            resfile_text += f"{resnum} {chain} PIKAA {mut}\n"

        with open(f"{output_dir}/nataa_mutations.resfile", "w") as f:
            f.write(resfile_text)

        # Write chains to move
        with open(f"{output_dir}/chains_to_move.txt", "w") as f:
            f.write("\n".join(row["chains_to_move"]))

        # Link PDB file
        pdbfile = os.path.abspath(row["pdbfile"])
        assert os.path.exists(pdbfile), f"PDB file not found: {pdbfile}"
        if not os.path.exists(f"{output_dir}/input.pdb"):
            os.symlink(pdbfile, f"{output_dir}/input.pdb")

    print("Done!")

if __name__ == "__main__":
    main()
