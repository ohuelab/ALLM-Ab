import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import os
import pandas as pd
from utils import load_config
from embeddings import blosum_embedding, ablang_embedding
from tqdm import tqdm

import pickle
def filtering_seqs(seq):
    # avoid free C
    return seq not in ["C"]


def mutseq_to_mut(mutseq, wt_seq, chain=None, offset=0, indel_indices=None):
    """Convert mutation sequence to list of mutations in standard format"""
    chain = chain or ""
    if indel_indices is None:
        indices2indel = {i:i+offset for i in range(len(wt_seq))}
    else:
        indices2indel = {i:v for i,v in enumerate(indel_indices)}

    mutations = []
    assert len(mutseq) == len(wt_seq)
    for i, (wt, mut) in enumerate(zip(wt_seq, mutseq)):
        if wt != mut:
            pos = indices2indel[i]
            mutations.append(f"{wt}{chain}{pos}{mut}")

    return mutations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--pool_data", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--bias", action="store_true")
    args = parser.parse_args()

    if os.path.exists(os.path.join(args.target_dir, "inference_result.csv")):
        print("Inference result already exists. Exiting...")
        exit()

    pool_df = pd.read_csv(args.pool_data)
    pool_df = pool_df[["mutseq", "mutations"]]
    print("Loading config and model...")
    config = load_config(args.config)
    generation_config = config["generation"]
    model_type = config.get("model_type", "sequence")
    assert model_type in ["ablang_gp", "blosum_gp"], f"Invalid model type: {model_type}"

    wt_seq = config["wildtype_sequence"][config["mutable_chain"]]
    mask_indices = np.array(config["mask_indices"])
    light_wt_seq = config["light_wt_sequence"]

    paired_seqs = []
    for mutseq in pool_df["mutseq"]:
        seq = list(wt_seq)
        for i, mut_aa in zip(mask_indices, mutseq):
            seq[i] = mut_aa
        paired_seqs.append([''.join(seq), light_wt_seq])

    if model_type == "ablang_gp":
        embeddings = ablang_embedding(paired_seqs)
    elif model_type == "blosum_gp":
        embeddings = np.array(list(map(blosum_embedding, tqdm(pool_df["mutseq"]))))

    with open(args.model_path, 'rb') as f:
        gp_model = pickle.load(f)

    preds, stds = gp_model.predict(embeddings, return_std=True)

    print("Creating results dataframe...")
    df = pd.DataFrame({
        "mutseq": pool_df["mutseq"],
        "mutations": pool_df["mutations"],
        "score": preds,
        "uncertainty": stds,
    })

    print("Sorting and filtering results...")
    sorted_df = df.sort_values(by="score", ascending=False).reset_index(drop=True)
    sorted_df = sorted_df.drop_duplicates(subset="mutseq").reset_index(drop=True)
    sorted_df = sorted_df[sorted_df["mutseq"].apply(filtering_seqs)].reset_index(drop=True)

    print("Saving results...")
    sorted_df.to_csv(os.path.join(args.target_dir, "generation_result.csv"), index=False)
    print(f"Generation result saved to {os.path.join(args.target_dir, 'generation_result.csv')}")
