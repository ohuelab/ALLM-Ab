import argparse
import numpy as np
import pandas as pd
from strategy import get_selector
from strategy_gp import get_selector as get_selector_gp
from utils import load_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--pool_data", type=str, required=True)
    parser.add_argument("--training_data", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    return parser.parse_args()

def main(args):
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

    if config.get("model_type", "sequence") in ["sequence", "ablang2"]:
        selector = get_selector(strategy_name, strategy_params)
        # Set best score for EI/PI strategies
        if strategy_name in ["ei", "pi"]:
            selector.best_score = train_df["DMS_score"].max() if train_df is not None else None

        selection_size = acquisition_config.get("acquisition_num", 50)
        print(f"Selecting {selection_size} samples using {strategy_name} strategy...")
        preds = pool_df["score"].values
        score_columns = [col for col in pool_df.columns if "score_ens_" in col]
        preds_list = np.array([pool_df[col].values for col in score_columns])

        if strategy_name in ["random", "greedy"]:
            selected_indices = selector.select_samples(pool_df, preds, selection_size)
        elif strategy_name in ["ei", "pi","ucb","thompson"]:
            selected_indices = selector.select_samples(pool_df, preds_list, selection_size)
    elif config.get("model_type", "sequence") in ["ablang_gp", "blosum_gp"]:
        if "uncertainty" not in pool_df.columns:
            score_columns = [col for col in pool_df.columns if "score_ens_" in col]
            preds_list = np.array([pool_df[col].values for col in score_columns])
            pool_df["uncertainty"] = np.std(preds_list, axis=0)

        selector = get_selector_gp(strategy_name, strategy_params)
        if strategy_name in ["ei", "pi"]:
            selector.best_score = train_df["DMS_score"].max() if train_df is not None else None
        selection_size = acquisition_config.get("acquisition_num", 50)
        print(f"Selecting {selection_size} samples using {strategy_name} strategy...")
        preds = pool_df["score"].values
        uncertainty = pool_df["uncertainty"].values
        if strategy_name in ["random", "greedy"]:
            selected_indices = selector.select_samples(pool_df, preds, selection_size)
        elif strategy_name in ["ei", "pi","ucb","thompson"]:
            selected_indices = selector.select_samples(pool_df, preds, uncertainty, selection_size)
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")


    # Save selected samples
    print(f"Saving {len(selected_indices)} selected samples to {args.output_file}")
    selected_df = pool_df.iloc[selected_indices]
    selected_df.to_csv(args.output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    args = parse_args()
    main(args)
