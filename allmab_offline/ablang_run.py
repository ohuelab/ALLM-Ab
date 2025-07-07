import os
import gc
import random
import copy
import yaml
import json
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, train_test_split
import scipy
# import ablang2

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


from strategy_gp import get_selector


DMS_COL = 'DMS_score'


def parse_config(config_path):
    """
    Reads YAML configuration for active learning and model settings.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed=42):
    """
    Sets random seeds for reproducibility across libraries and environments.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True


def generate_unique_id(string):
    """
    Generates a unique SHA-1 hash from a given string.
    """
    import hashlib
    hash_object = hashlib.sha1()
    hash_object.update(string.encode('utf-8'))
    return hash_object.hexdigest()


def write_log(log_file, text, is_print=True):
    """
    Writes text to a log file and optionally prints it.
    """
    if is_print:
        print(text)
    datetime = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{datetime}] {text}\n")
    log_file.flush()


def metric(preds, labels, bottom_preds=None, bottom_labels=None,
           top_preds=None, top_labels=None, eps=1e-6):
    """
    Calculates performance metrics (Pearson, Spearman, RMSE).
    If subsets (top/bottom) are provided, also calculates precision/recall/F1.
    """
    pearson = scipy.stats.pearsonr(preds, labels)[0]
    spearman = scipy.stats.spearmanr(preds, labels)[0]
    rmse = np.mean((preds - labels) ** 2) ** 0.5
    ms = {
        'pearson': pearson,
        'spearman': spearman,
        'rmse': rmse
    }

    if top_preds is not None and bottom_preds is not None:
        all_preds = np.concatenate([bottom_preds, preds, top_preds])
        n_top = len(top_preds)
        n_bottom = len(bottom_preds)
        top_pred_idxs = np.argsort(all_preds)[-n_top:]
        top_metrics = []
        bottom_metrics = []
        unbias_metrics = []

        for k in [10, 20, 50, 100]:
            precision = (top_pred_idxs[-min(k, n_top):] >= (len(all_preds) - n_top)).mean()
            recall = (top_pred_idxs[-min(k, n_top):] >= (len(all_preds) - n_top)).sum() / n_top
            f1 = 2 * precision * recall / (precision + recall + eps)
            top_metrics.append([precision, recall, f1])

            precision = (top_pred_idxs[:min(k, n_bottom)] < n_bottom).mean()
            recall = (top_pred_idxs[:min(k, n_bottom)] < n_bottom).sum() / n_top
            f1 = 2 * precision * recall / (precision + recall + eps)
            bottom_metrics.append([precision, recall, f1])

        precision = (top_pred_idxs >= (len(all_preds) - n_top)).mean()
        recall = (top_pred_idxs >= (len(all_preds) - n_top)).sum() / n_top
        f1 = 2 * precision * recall / (precision + recall + eps)
        top_metrics.append([precision, recall, f1])

        precision = (top_pred_idxs < n_bottom).mean()
        recall = (top_pred_idxs < n_bottom).sum() / n_top
        f1 = 2 * precision * recall / (precision + recall + eps)
        bottom_metrics.append([precision, recall, f1])

        for i, _ in enumerate([10, 20, 50, 100, 'top_frac']):
            unbias_metrics.append(
                list(np.array(top_metrics[i]) - np.array(bottom_metrics[i]))
            )

        ms.update({
            'top10_precision_recall_f1': top_metrics[0],
            'top20_precision_recall_f1': top_metrics[1],
            'top50_precision_recall_f1': top_metrics[2],
            'top100_precision_recall_f1': top_metrics[3],
            'top_frac_precision_recall_f1': top_metrics[4],

            'bottom10_precision_recall_f1': bottom_metrics[0],
            'bottom20_precision_recall_f1': bottom_metrics[1],
            'bottom50_precision_recall_f1': bottom_metrics[2],
            'bottom100_precision_recall_f1': bottom_metrics[3],
            'bottom_frac_precision_recall_f1': bottom_metrics[4],

            'unbias10_precision_recall_f1': unbias_metrics[0],
            'unbias20_precision_recall_f1': unbias_metrics[1],
            'unbias50_precision_recall_f1': unbias_metrics[2],
            'unbias100_precision_recall_f1': unbias_metrics[3],
            'unbias_frac_precision_recall_f1': unbias_metrics[4],
        })

    return ms

def check_config(config):
    """
    Checks the configuration settings for consistency.
    """
    if config["active_learning"]["strategy"] in ["ei", "pi", "ucb", "thompson"]:
        assert config.get("use_dropout", False) or config.get("ensemble", False), \
            "Ensemble-based selection strategies require dropout sampling or cv models."

from sklearn.gaussian_process.kernels import Kernel
class TanimotoKernel(Kernel):
    """
    Tanimoto kernel implementation.

    The Tanimoto kernel is defined as:
    k(x,y) = <x,y> / (<x,x> + <y,y> - <x,y>)
    """
    def __init__(self):
        super().__init__()

    def __call__(self, X, Y=None):
        if Y is None:
            Y = X

        # Convert to arrays
        X = np.asarray(X)
        Y = np.asarray(Y)

        # Calculate dot products
        xy = np.dot(X, Y.T)
        xx = np.sum(X * X, axis=1)
        yy = np.sum(Y * Y, axis=1)

        # Reshape for broadcasting
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(1, -1)

        # Calculate Tanimoto kernel
        K = xy / (xx + yy - xy)

        return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X)."""
        return np.ones(X.shape[0])

    def is_stationary(self):
        """Returns whether the kernel is stationary."""
        return False


def get_predictions(train_df, X_all, X_train, config):
    dms_column=DMS_COL if config["noise_level"] == 0 else DMS_COL + "_noise"
    y_train = train_df[dms_column].tolist()
    kernel_type = config.get("kernel", "rbf")
    if kernel_type == "rbf":
        kernel = RBF()
    elif kernel_type == "matern":
        # mattern-3/2
        kernel = Matern(nu=1.5)
    elif kernel_type == "tanimoto":
        kernel = TanimotoKernel()
    else:
        raise ValueError(f"Invalid kernel: {kernel_type}")
    if config.get("add_kernel", False):
        kernel = kernel + WhiteKernel() + ConstantKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)
    gpr.fit(X_train, y_train)
    return gpr.predict(X_all, return_std=True)


def main(config):
    """
    Main function for the active learning experiment.
    Separates out training logic and passes model predictions
    (rather than the model) to the selection strategy.
    """
    seed_everything(config["random_seed"])

    # Log file
    os.makedirs(config["tmp_path"], exist_ok=True)
    logfile_path = os.path.join(config["tmp_path"], 'active_learning_log.txt')
    logfile = open(logfile_path, 'a')
    if os.path.exists(logfile_path):
        logfile.write('\n\n')
        write_log(logfile, f"Resuming active learning experiment with config: {config}")

    # Load mapping and select DMS data
    train_mapping_path = config["train_dms_mapping"]
    train_df_mapping = pd.read_csv(train_mapping_path)
    name = train_df_mapping.loc[config["dms_index"], 'DMS_id']

    # Active learning parameters
    check_config(config)
    N_init = config["active_learning"]["N_init"]
    N_per_cycle = config["active_learning"]["N_per_cycle"]
    M_cycles = config["active_learning"]["M_cycles"]
    strategy_params = config["active_learning"].get("strategy_params", {})
    strategy_name = config["active_learning"]["strategy"]
    if strategy_name in ["random", "thompson"]:
        strategy_params["seed"] = config["random_seed"]

    initial_selection_mode = config.get("initial_selection_mode", "random")

    # Load data and initialize cycle
    initial_cycle = 0
    for cycle in range(M_cycles):
        cycle_df_path = os.path.join(config["tmp_path"], f"predictions_cycle_{cycle+1}.csv")
        if os.path.exists(cycle_df_path):
            df_all = pd.read_csv(cycle_df_path)
            initial_cycle = cycle + 1
    write_log(logfile, f"Starting from cycle {initial_cycle}")
    # ablang = ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device='cpu')

    X_all = np.load(config["embeddings"])


    if initial_cycle == 0:
        write_log(logfile, "Starting new active learning experiment.")
        dms_data_path = os.path.join(config["dms_input"], f"{name}.csv")
        df_all = pd.read_csv(dms_data_path)
        df_all["selected_cycle"] = -1
        df_all["is_test"] = False

        all_seqs = list(zip(df_all["heavy"].tolist(), df_all["light"].tolist()))
        # X_all = ablang(all_seqs, mode='seqcoding')
        # np.save(os.path.join(config["tmp_path"], "X_all.npy"), X_all)

        # Split test data
        test_df = df_all.sample(n=config["test_size"], random_state=config["random_seed"])
        test_indices = test_df.index.tolist()
        pool_df = df_all.drop(test_df.index)
        pool_indices = pool_df.index.tolist()
        all_indices = {"pool": pool_indices, "test": test_indices}
        df_all.loc[test_indices, "is_test"] = True
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "w") as f:
            json.dump(all_indices, f)

        # Optionally add noise column
        if config["noise_level"] > 0 and DMS_COL+"_noise" not in pool_df.columns:
            write_log(logfile, f"  Adding noise to DMS scores with std = {config['noise_level']}")
            rng = np.random.RandomState(config["random_seed"])
            dstd = pool_df[DMS_COL].std() * config["noise_level"]
            pool_df[DMS_COL + "_noise"] = pool_df[DMS_COL] + rng.normal(0, dstd, size=len(pool_df))

        # 初期サンプリングの分岐  # CHANGED
        if initial_selection_mode == "random":
            write_log(logfile, f"  Initializing with {N_init} random samples.")
            initial_selector = get_selector("random", {"seed": config["random_seed"]})
            init_indices = initial_selector.select_samples(pool_df, None, N_init)
        else:
            raise ValueError("ERROR")
        # df_all.to_csv(os.path.join(config["tmp_path"], "predictions_cycle_0.csv"), index=False)

        train_selected_df = pool_df.loc[init_indices].copy()
        X_train = X_all[init_indices]
        df_all.loc[init_indices, "selected_cycle"] = 0
        pool_df.drop(init_indices, inplace=True)
        all_indices[str(0)] = init_indices
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "w") as f:
            json.dump(all_indices, f)

    else:
        # X_all = np.load(os.path.join(config["tmp_path"], "X_all.npy"))
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "r") as f:
            all_indices = json.load(f)
        initial_pool_indices = all_indices["pool"]
        test_indices = all_indices["test"]
        test_df = df_all.loc[test_indices]
        train_indices = []
        for cycle in range(initial_cycle+1):
            if str(cycle) in all_indices:
                train_indices.extend(all_indices[str(cycle)])
        pool_df = df_all.loc[initial_pool_indices].copy()
        if config["noise_level"] > 0 and DMS_COL+"_noise" not in pool_df.columns:
            write_log(logfile, f"  Adding noise to DMS scores with std = {config['noise_level']}")
            rng = np.random.RandomState(config["random_seed"])
            dstd = pool_df[DMS_COL].std() * config["noise_level"]
            pool_df[DMS_COL + "_noise"] = pool_df[DMS_COL] + rng.normal(0, dstd, size=len(pool_df))

        train_selected_df = pool_df.loc[train_indices].copy()
        X_train = X_all[train_indices]
        pool_df.drop(train_indices, inplace=True)
        assert (df_all["selected_cycle"]>=0).sum() == len(train_indices), \
            "Mismatch in selected cycle indices: {} vs {}".format((df_all["selected_cycle"]>=0).sum(), len(train_indices))
    assert len(X_all)==len(df_all)

    for cycle in range(initial_cycle, M_cycles):
        # Re-initialize selector
        selector = get_selector(strategy_name, strategy_params)
        best_score = train_selected_df[DMS_COL+"_noise"].max() if config["noise_level"] > 0 else train_selected_df[DMS_COL].max()
        if strategy_name in ["ei", "pi"]:
            selector.best_score = best_score

        write_log(logfile, f"\n=== Active Learning Cycle {cycle+1} / {M_cycles} ===")


        all_preds, all_uncertainties = get_predictions(train_selected_df, X_all, X_train, config)
        df_all[f"cycle_{cycle+1}_preds"] = all_preds
        df_all[f"cycle_{cycle+1}_uncertainties"] = all_uncertainties

        # Active selection from pool
        if len(pool_df) > 0:
            pool_preds = all_preds[pool_df.index]
            pool_uncertainties = all_uncertainties[pool_df.index]
            if strategy_name in ["ei", "pi", "ucb", "thompson"]:
                add_indices = selector.select_samples(pool_df,  mu=pool_preds, sigma=pool_uncertainties, selection_size=N_per_cycle)
            else:
                add_indices = selector.select_samples(pool_df, preds=pool_preds, selection_size=N_per_cycle)

            new_data = pool_df.loc[add_indices]
            train_selected_df = pd.concat([train_selected_df, new_data]).reset_index(drop=True)
            X_train = np.concatenate([X_train, X_all[add_indices]])
            df_all.loc[add_indices, "selected_cycle"] = cycle + 1
            assert len(X_train) == len(train_selected_df), "Mismatch in X_train and train_selected_df"
            pool_df.drop(add_indices, inplace=True)
        else:
            write_log(logfile, "  No more samples to select from pool.")

        # Save CSV with selection info

        # Save indices
        all_indices[str(cycle + 1)] = add_indices
        with open(os.path.join(config["tmp_path"], "selected_indices.json"), "w") as f:
            json.dump(all_indices, f)
        if len(pool_df) == 0:
            break
    pred_csv_path = os.path.join(config["tmp_path"], f"predictions_cycle_{cycle+1}.csv")
    df_all.to_csv(pred_csv_path, index=False)

    logfile.close()


if __name__ == '__main__':
    """
    Example usage:
    python active_learning_script.py --config config.yaml
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=-1)
    args_cmd = parser.parse_args()
    config_dict = parse_config(args_cmd.config)
    config_dict["batch_size"] = args_cmd.batch_size
    if args_cmd.seed != -1:
        config_dict["random_seed"] = args_cmd.seed
    main(config_dict)
