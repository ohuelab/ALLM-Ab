import argparse
import numpy as np
import pandas as pd
from strategy_gp import get_selector
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
    parser.add_argument("--training_datas", type=str, nargs="+")
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()

# def main(args):
if __name__ == "__main__":
    args = parse_args()
    print("Loading config and data...")
    config = load_config(args.config)
    acquisition_config = config["acquisition"]

    print(f"Loading pool data from {args.pool_data}")
    pool_df = pd.read_csv(args.pool_data)
    # Load training and generation data
    if args.training_datas is None:
        args.training_datas = []
    for training_data in args.training_datas:
        print(f"Loading training data from {training_data}")
        train_df = pd.read_csv(training_data)
        pool_df = pool_df[~pool_df["mutseq"].isin(train_df["mutseq"])].reset_index(drop=True)
    print("Setting up selector...")

    num_score = 2

    selectors = []
    for i in range(num_score):
        strategy_name = acquisition_config["strategy_name"]
        strategy_params = acquisition_config.get("strategy_params", {})
        selector = get_selector(strategy_name, strategy_params)
        selectors.append(selector)
        # Set best score for EI/PI strategies
        if strategy_name in ["ei", "pi"]:
            raise NotImplementedError("EI/PI strategy is not implemented for multi-objective hypervolume selection")

    selection_size = acquisition_config.get("acquisition_num", 50)
    print(f"Selecting {selection_size} samples using with multi-objective hypervolume...")

    preds_list = []
    uncertainties_list = []
    for i in range(num_score):
        preds_list.append(pool_df[f"multi_score_{i}"].values)
        if "uncertainty" in pool_df.columns:
            uncertainties_list.append(pool_df["uncertainty"])
        else:
            score_columns = [col for col in pool_df.columns if f"multi_score_ens_{i}" in col]
            preds = np.array([pool_df[col].values for col in score_columns])
            uncertainties = np.std(preds, axis=0)
            uncertainties_list.append(uncertainties)
    preds = np.array(preds_list)
    uncertainties = np.array(uncertainties_list)
    for i in range(num_score):
        selector = selectors[i]
        if strategy_name in ["random", "greedy"]:
            acquisition_scores = selector.acquisition_score(preds[i])
        elif strategy_name in ["ei", "pi","ucb","thompson"]:
            acquisition_scores = selector.acquisition_score(preds[i], uncertainties[i])
        pool_df[f"acquisition_score_{i}"] = acquisition_scores

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
    score_cols = []
    acquisition_weight = config.get("acquisition_weight", {})
    print(acquisition_weight,config)
    for i in range(num_score):
        pool_df[f"acquisition_score_std_{i}"] = normalize_score(-pool_df[f"acquisition_score_{i}"])
        print("acquisition_score_",acquisition_weight.get(f"acquisition_score_{i}", 2))
        if acquisition_weight.get(f"acquisition_score_{i}", 2) > 0:
            score_cols.append(f"acquisition_score_std_{i}")
    pool_df["ablang2_perplexity_std"] = normalize_score(pool_df["ablang2_perplexity"])
    if acquisition_weight.get("ablang2_perplexity", 1) > 0:
        score_cols.append("ablang2_perplexity_std")
    pool_df["IP_seq_std"] = normalize_score(-pool_df["IP_seq"])
    if acquisition_weight.get("IP_seq", 1) > 0:
        score_cols.append("IP_seq_std")

    for i in range(num_score):
        pool_df[f"acquisition_score_std_{i}"] *= acquisition_weight.get(f"acquisition_score_{i}", 2)
    pool_df["ablang2_perplexity_std"] *= acquisition_weight.get("ablang2_perplexity", 1)
    pool_df["IP_seq_std"] *= acquisition_weight.get("IP_seq", 1)

    ref_point = []
    for col in score_cols:
        ref_point.append(pool_df[col].quantile(0.95))

    Nprev = len(pool_df)
    pool_df_0 = pool_df.copy()
    print("score_cols", score_cols)
    assert len(ref_point) == len(score_cols), "ref_point length is not equal to score_cols length"
    for i in range(len(score_cols)):
        pool_df = pool_df[pool_df[score_cols[i]] <= ref_point[i]]
    pool_df[pool_df["ablang2_perplexity"]<config.acquisition.get("ablang2_perplexity_thr", np.inf)]
    Nnew = len(pool_df)
    print(f"Selected {Nnew} samples from {Nprev} pool data")

    X = pool_df[score_cols].values
    multi_objective_strategy = config.acquisition.get("multi_objective_strategy", "hypervolume")
    if multi_objective_strategy == "hypervolume":
        selected_indices = greedy_hypervolume_subset(X, selection_size, ref_point)
    elif multi_objective_strategy == "sum":
        pool_df["sum_score"] = pool_df[score_cols].sum(axis=1)
        selected_indices = pool_df.sort_values("sum_score", ascending=True).index[:selection_size]
    elif multi_objective_strategy == "non_dominated":
        from fast_pareto import is_pareto_front, nondominated_rank
        ranks = nondominated_rank(X)
        selected_indices = pool_df.iloc[np.argsort(ranks)].index[:selection_size]
    else:
        raise ValueError(f"Invalid multi-objective strategy: {multi_objective_strategy}")
    selected_indices = pool_df.index[selected_indices]

    selected_df = pool_df.loc[selected_indices]

    # Save selected samples
    print(f"Saving {len(selected_indices)} selected samples to {args.output_file}")
    selected_df.to_csv(args.output_file, index=False)
    pool_df.to_csv(args.pool_data.replace(".csv", "_multi_objective.csv"), index=False)
    print("Done!")
