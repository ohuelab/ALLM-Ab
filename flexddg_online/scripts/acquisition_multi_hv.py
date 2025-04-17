import argparse
import numpy as np
import pandas as pd
from strategy import get_selector
from utils import load_config

from logits import ablang2_perplexity
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from ablang2.load_model import load_model

from tqdm import tqdm
from pygmo import hypervolume

def greedy_hypervolume_subset(points, n, ref_point):
    selected = []
    remaining = list(range(len(points)))

    for _ in tqdm(range(n)):
        max_hv = -float('inf')
        best_idx = None

        for idx in remaining:
            # 現在の選択 + 候補点のHypervolume計算
            current_points = points[selected + [idx]]
            hv = hypervolume(current_points)
            current_hv = hv.compute(ref_point)

            if current_hv > max_hv:
                max_hv = current_hv
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected

def calculate_IP(sequence):
    return ProteinAnalysis(sequence).isoelectric_point()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pool_data", type=str, required=True)
    parser.add_argument("--training_data", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()

# def main(args):
if __name__ == "__main__":
    args = parse_args()
    print("Loading config and data...")
    config = load_config(args.config)
    acquisition_config = config["acquisition"]

    # Load training and generation data
    if args.training_data is not None:
        print(f"Loading training data from {args.training_data}")
        train_df = pd.read_csv(args.training_data)
    else:
        train_df = None
    print(f"Loading pool data from {args.pool_data}")
    pool_df = pd.read_csv(args.pool_data)
    # subtract sample from pool if it is in training data
    if train_df is not None:
        pool_df = pool_df[~pool_df["mutseq"].isin(train_df["mutseq"])].reset_index(drop=True)
    print("Setting up selector...")
    strategy_name = acquisition_config["strategy_name"]
    strategy_params = acquisition_config.get("strategy_params", {})
    selector = get_selector(strategy_name, strategy_params)

    # Set best score for EI/PI strategies
    if strategy_name in ["ei", "pi"]:
        selector.best_score = train_df["DMS_score"].max() if train_df is not None else None

    selection_size = acquisition_config.get("acquisition_num", 50)
    print(f"Selecting {selection_size} samples using with multi-objective hypervolume...")
    preds = pool_df["score"].values
    if "uncertainty" in pool_df.columns:
        uncertainties = pool_df["uncertainty"]
    else:
        score_columns = [col for col in pool_df.columns if "score_ens_" in col]
        preds_list = np.array([pool_df[col].values for col in score_columns])
        uncertainties = np.std(preds_list, axis=0)

    if strategy_name in ["random", "greedy"]:
        acquisition_scores = selector.acquisition_score(preds)
    elif strategy_name in ["ei", "pi","ucb","thompson"]:
        acquisition_scores = selector.acquisition_score(preds, uncertainties)
    pool_df["acquisition_score"] = acquisition_scores

    # Only applicabele for antibody CDR-H3
    print("Calculating ablang2 perplexity and IP...")
    wildtype_sequence = config["wildtype_sequence"][config["mutable_chain"]]
    light_sequence = config["light_wt_sequence"]
    mask_indices = config["mask_indices"]

    mutated_sequences=[]
    paired_sequences = []
    for i, d in pool_df.iterrows():
        mutseq=d["mutseq"]
        mutated_sequence = "".join([wildtype_sequence[i] if i not in mask_indices else mutseq[mask_indices.index(i)] for i in range(len(wildtype_sequence))])
        mutated_sequences.append(mutated_sequence)
        paired_sequences.append([mutated_sequence, light_sequence])

    masked_wildtype_sequence = "".join([wildtype_sequence[i] if i not in mask_indices else "*" for i in range(len(wildtype_sequence))])
    masked_pair_seq = [masked_wildtype_sequence, light_sequence]
    AbLang, tokenizer, _ = load_model("ablang2-paired")
    pool_df["ablang2_perplexity"] = ablang2_perplexity(paired_sequences, AbLang, tokenizer, masked_pair_seq, mask_indices, device="cpu")
    pool_df["IP_seq"] = list(map(calculate_IP, mutated_sequences))


    # Hypervolume selection
    print("Selecting Hypervolume...")
    def normalize_score(score):
        return (score-score.quantile(0.05))/(score.quantile(0.95)-score.quantile(0.05)+1e-10)
    pool_df["acquisition_score_std"] = normalize_score(-pool_df["acquisition_score"])
    pool_df["ablang2_perplexity_std"] = normalize_score(pool_df["ablang2_perplexity"])
    pool_df["IP_seq_std"] = normalize_score(-pool_df["IP_seq"])
    acquisition_weight = config.get("acquisition_weight", {})
    pool_df["acquisition_score_std"] *= acquisition_weight.get("acquisition_score", 2)
    pool_df["ablang2_perplexity_std"] *= acquisition_weight.get("ablang2_perplexity", 1)
    pool_df["IP_seq_std"] *= acquisition_weight.get("IP_seq", 1)

    acquisition_score_ref = pool_df["acquisition_score_std"].quantile(0.95)
    perplexity_ref = pool_df["ablang2_perplexity_std"].quantile(0.95)
    ip_seq_ref = pool_df["IP_seq_std"].quantile(0.95)

    Nprev = len(pool_df)
    pool_df_0 = pool_df.copy()
    pool_df = pool_df[pool_df["acquisition_score_std"] <= acquisition_score_ref]
    pool_df = pool_df[pool_df["ablang2_perplexity_std"] <= perplexity_ref]
    pool_df = pool_df[pool_df["IP_seq_std"] <= ip_seq_ref]
    pool_df[pool_df["ablang2_perplexity"]<config.acquisition.get("ablang2_perplexity_thr", np.inf)]
    pre_cols = ["acquisition_score", "ablang2_perplexity", "IP_seq"]
    obj_cols = []
    ref_point = []
    for col in pre_cols:
        if acquisition_weight.get(col, 1) > 0:
            obj_cols.append(col+"_std")
            if "acquisition_score" in col:
                ref_point.append(acquisition_weight.get(col, 2))
            else:
                ref_point.append(acquisition_weight.get(col, 1))
    print("ref_point",ref_point)
    pool_df = pool_df.sort_values(obj_cols, ascending=True)
    Nnew = len(pool_df)
    print(f"Selected {Nnew} samples from {Nprev} pool data")
    X = pool_df[obj_cols].values
    multi_objective_strategy = config.acquisition.get("multi_objective_strategy", "hypervolume")
    if multi_objective_strategy == "hypervolume":
        selected_indices = greedy_hypervolume_subset(X, selection_size, ref_point)
    elif multi_objective_strategy == "sum":
        # select top order of sum of scores, smaller is better
        pool_df["sum_score"] = pool_df["acquisition_score_std"] + pool_df["ablang2_perplexity_std"] + pool_df["IP_seq_std"]
        pool_df = pool_df.sort_values("sum_score", ascending=True)
        selected_indices = pool_df.index[:selection_size]
    elif multi_objective_strategy == "non_dominated":
        from fast_pareto import is_pareto_front, nondominated_rank
        ranks = nondominated_rank(X)
        pool_df = pool_df.iloc[np.argsort(ranks)]
        selected_indices = pool_df.index[:selection_size]
    else:
        raise ValueError(f"Invalid multi-objective strategy: {multi_objective_strategy}")
    selected_indices = pool_df.index[selected_indices]

    selected_df = pool_df.loc[selected_indices]

    # Save selected samples
    print(f"Saving {len(selected_indices)} selected samples to {args.output_file}")
    selected_df.to_csv(args.output_file, index=False)
    pool_df.to_csv(args.pool_data.replace(".csv", "_multi_objective.csv"), index=False)
    print("Done!")

    # main(args)
